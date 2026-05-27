use anyhow::{Context, Result};
use coreaudio_sys::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::ffi::CStr;
use std::io::{self, BufRead, Write};
use std::mem;
use std::process::Command;
use std::sync::{Arc, Mutex};

const CHANNELS: u16 = 2;
const RING_BUFFER_MS: usize = 1000;

const LOOPBACK_KEYWORDS: &[&str] = &[
    "blackhole", "loopback", "soundflower",
    "eqmac", "menubus", "virtual", "wavtap",
];

struct MixerState {
    mic_ring: Vec<f32>,
    mic_write: usize,
    
    // Main output
    main_read: usize,
    main_count: usize,
    main_started: bool,
    
    // Monitor output
    mon_read: usize,
    mon_count: usize,
    mon_started: bool,
    
    tts_buffer: Vec<f32>,
    tts_main_idx: usize,
    tts_mon_idx: usize,
    
    sample_rate: u32,
    input_channels: u16,
    mic_signal: f32,
    out_signal: f32,
    tts_only_monitor: bool,
}

impl MixerState {
    fn new(sample_rate: u32, input_channels: u16, tts_only_monitor: bool) -> Self {
        let size = (sample_rate as usize) * (CHANNELS as usize) * RING_BUFFER_MS / 1000;
        Self {
            mic_ring: vec![0.0; size],
            mic_write: 0,
            main_read: 0,
            main_count: 0,
            main_started: false,
            mon_read: 0,
            mon_count: 0,
            mon_started: false,
            tts_buffer: Vec::new(),
            tts_main_idx: 0,
            tts_mon_idx: 0,
            sample_rate,
            input_channels,
            mic_signal: 0.0,
            out_signal: 0.0,
            tts_only_monitor,
        }
    }

    fn push_input(&mut self, samples: &[f32]) {
        let len = self.mic_ring.len();
        for chunk in samples.chunks_exact(self.input_channels as usize) {
            let mono = chunk[0];
            for _ in 0..CHANNELS {
                self.mic_ring[self.mic_write] = mono;
                self.mic_write = (self.mic_write + 1) % len;
                
                if self.main_count < len { self.main_count += 1; }
                else { self.main_read = (self.main_read + 1) % len; }
                
                if self.mon_count < len { self.mon_count += 1; }
                else { self.mon_read = (self.mon_read + 1) % len; }
            }
            self.mic_signal = self.mic_signal.max(mono.abs());
        }
    }

    fn fill_main(&mut self, data: &mut [f32]) {
        let threshold = (self.sample_rate as usize * CHANNELS as usize) / 20;
        if !self.main_started && self.main_count >= threshold { self.main_started = true; }

        for sample in data.iter_mut() {
            let mut val = 0.0;
            if self.main_started && self.main_count > 0 {
                val = self.mic_ring[self.main_read];
                self.main_read = (self.main_read + 1) % self.mic_ring.len();
                self.main_count -= 1;
            }
            if self.tts_main_idx < self.tts_buffer.len() {
                val += self.tts_buffer[self.tts_main_idx];
                self.tts_main_idx += 1;
            }
            *sample = val;
            self.out_signal = self.out_signal.max(val.abs());
        }
    }

    fn fill_mon(&mut self, data: &mut [f32]) {
        let threshold = (self.sample_rate as usize * CHANNELS as usize) / 20;
        if !self.mon_started && self.mon_count >= threshold { self.mon_started = true; }

        for sample in data.iter_mut() {
            let mut val = 0.0;
            if !self.tts_only_monitor && self.mon_started && self.mon_count > 0 {
                val = self.mic_ring[self.mon_read];
                self.mon_read = (self.mon_read + 1) % self.mic_ring.len();
                self.mon_count -= 1;
            }
            if self.tts_mon_idx < self.tts_buffer.len() {
                val += self.tts_buffer[self.tts_mon_idx];
                self.tts_mon_idx += 1;
            }
            *sample = val;
        }
    }
}

fn main() -> Result<()> {
    let tts_only_monitor = std::env::args().any(|a| a == "--tts-only-monitor" || a == "-tom");
    let enable_monitor = tts_only_monitor || std::env::args().any(|a| a == "--monitor" || a == "-m");
    let auto_install = std::env::args().any(|a| a == "--install" || a == "-i");
    let speaker_mode = std::env::args().any(|a| a == "--speaker" || a == "-s");

    println!("\n=== Rust Virtual Audio Mixer ===");

    let host = cpal::default_host();

    let input_device = select_device(host.input_devices()?, "Input (Microphone)", "MacBook Pro Microphone")?;

    let output_device = if speaker_mode {
        host.default_output_device().context("No output device available")?
    } else {
        select_output_device(&host, auto_install)?
    };

    // warn if output is not a loopback device (other apps won't hear it)
    if let Ok(desc) = output_device.description() {
        let name = desc.name().to_owned();
        let is_loopback = LOOPBACK_KEYWORDS.iter().any(|k| name.to_lowercase().contains(k));
        if !is_loopback {
            eprintln!("[WARN] \"{}\" is not a loopback/virtual device.", name);
            eprintln!("[WARN] Other apps (Zoom, Discord, etc.) won't hear the mixed audio.");
            eprintln!("[WARN] Select a loopback device (BlackHole, etc.) or use --install to install one.");
        } else {
            prompt_set_default_input(&name)?;
        }
    }

//    print!("\nEnable monitoring to default speaker? [y/N]: ");
//    io::stdout().flush()?;
//    let mut mon_input = String::new();
//    io::stdin().read_line(&mut mon_input)?;
//    let monitoring_enabled = mon_input.trim().to_lowercase() == "y";

    let config = output_device.default_output_config()?.config();
    let sample_rate = config.sample_rate;
    let mut input_config = input_device.default_input_config()?.config();
    let mut input_channels = input_config.channels;
    println!("[CONFIG] Output: {} Hz, Input default: {} Hz ({} channels)", sample_rate, input_config.sample_rate, input_channels);

    // Match input sample rate to output to prevent ring buffer drift/static
    if sample_rate != input_config.sample_rate {
        if let Ok(configs) = input_device.supported_input_configs() {
            for c in configs {
                if c.sample_format() == cpal::SampleFormat::F32
                    && c.min_sample_rate() <= sample_rate
                    && c.max_sample_rate() >= sample_rate
                {
                    if let Some(matched) = c.try_with_sample_rate(sample_rate) {
                        input_config = matched.config();
                        input_channels = input_config.channels;
                        println!("[CONFIG] Matched input to output sample rate: {} Hz", sample_rate);
                        break;
                    }
                }
            }
        }
    }
    if sample_rate != input_config.sample_rate {
        eprintln!("[WARN] Input sample rate ({}) differs from output ({}). Ring buffer may drift.",
            input_config.sample_rate, sample_rate);
    }

    let state = Arc::new(Mutex::new(MixerState::new(sample_rate, input_channels, tts_only_monitor)));

    let play_state = Arc::clone(&state);
    let _output_stream = output_device.build_output_stream(
        &config,
        move |data: &mut [f32], _| play_state.lock().unwrap().fill_main(data),
        |err| eprintln!("[Error] Output: {}", err),
        None
    )?;
    _output_stream.play()?;

    let mut monitor_stream = None;

    let capture_state = Arc::clone(&state);
    let capture_stream = input_device.build_input_stream(
        &input_config,
        move |data: &[f32], _| capture_state.lock().unwrap().push_input(data),
        |err| eprintln!("[Error] Input: {}", err),
        None
    )?;
    capture_stream.play()?;

    if enable_monitor {
        monitor_stream = start_monitor(&host, &state);
    }

    println!("\n[SYSTEM ACTIVE] Type for TTS or 'exit' to quit.");
    print!("> ");
    io::stdout().flush()?;
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let text = line?.trim().to_string();
        if text == "exit" { break; }
        if text == "toggle_monitoring" {
            if monitor_stream.is_some() {
                monitor_stream = None;
                println!("[MONITOR] Deactivated");
            } else {
                monitor_stream = start_monitor(&host, &state);
            }
            continue;
        }
        if text == "tts_only_monitor" {
            let mut s = state.lock().unwrap();
            s.tts_only_monitor = !s.tts_only_monitor;
            println!("[MONITOR] Mode: {}", if s.tts_only_monitor { "TTS only" } else { "Mic + TTS" });
            continue;
        }
        if !text.is_empty() {
            if let Err(e) = play_tts(&text, Arc::clone(&state)) {
                eprintln!("[Error] TTS: {}", e);
            }
        }
        print!("> ");
        io::stdout().flush()?;
    }
    Ok(())
}

fn select_device(devices: impl Iterator<Item = cpal::Device>, prompt: &str, default_match: &str) -> Result<cpal::Device> {
    let list: Vec<_> = devices.collect();
    println!("\n--- {} ---", prompt);
    let mut default_idx = None;
    for (i, d) in list.iter().enumerate() {
        let name = d.description().map(|desc| desc.name().to_owned()).unwrap_or_else(|_| "Unknown".into());
        let matches = name.contains(default_match);
        if matches { default_idx = Some(i); }
        println!("[{}] {}{}", i, name, if matches { " (Default)" } else { "" });
    }
    print!("Select index{}: ", default_idx.map(|i| format!(" [Enter for {}]", i)).unwrap_or_default());
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();
    if input.is_empty() {
        if let Some(i) = default_idx {
            return list.into_iter().nth(i).context("Default device not found");
        }
    }
    let idx: usize = input.parse().context("Invalid index")?;
    list.into_iter().nth(idx).context("Device out of range")
}

fn select_output_device(host: &cpal::Host, auto_install: bool) -> Result<cpal::Device> {
    let devices: Vec<_> = host.output_devices()?.collect();

    // auto-detect loopback device
    let mut loopback_idx = None;
    for (i, d) in devices.iter().enumerate() {
        if let Ok(desc) = d.description() {
            let name = desc.name();
            let lower = name.to_lowercase();
            if LOOPBACK_KEYWORDS.iter().any(|k| lower.contains(k)) {
                loopback_idx = Some(i);
                break;
            }
        }
    }

    let prompt = if loopback_idx.is_some() {
        "Output (Loopback / Virtual Device)"
    } else {
        "Output Device"
    };

    // show devices
    println!("\n--- {} ---", prompt);
    for (i, d) in devices.iter().enumerate() {
        let name = d.description().map(|desc| desc.name().to_owned()).unwrap_or_else(|_| "Unknown".into());
        let is_loopback = LOOPBACK_KEYWORDS.iter().any(|k| name.to_lowercase().contains(k));
        let tag = if Some(i) == loopback_idx { " (Default)" } else if is_loopback { " (Loopback)" } else { "" };
        println!("[{}] {}{}", i, name, tag);
    }

    // if no loopback device, offer to install
    if loopback_idx.is_none() && (auto_install || should_install_blackhole()?) {
        loopback_idx = install_blackhole()?;
    }

    let default_hint = loopback_idx.map(|i| format!(" [Enter for {}]", i)).unwrap_or_default();
    print!("Select index{}: ", default_hint);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();
    if input.is_empty() {
        if let Some(i) = loopback_idx {
            return Ok(devices.into_iter().nth(i).context("Device not found")?);
        }
    }
    if !input.is_empty() {
        let idx: usize = input.parse().context("Invalid index")?;
        return devices.into_iter().nth(idx).context("Device out of range");
    }

    // fallback: default output device (speakers)
    host.default_output_device().context("No output device available")
}

fn should_install_blackhole() -> Result<bool> {
    print!("\nNo loopback device found. Install BlackHole 2ch via Homebrew? [Y/n]: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_lowercase() != "n")
}

fn install_blackhole() -> Result<Option<usize>> {
    println!("[INSTALL] Installing BlackHole 2ch via Homebrew...");
    println!("[INSTALL] This may require your admin password for the macOS installer.");
    let status = Command::new("brew")
        .args(&["install", "--cask", "blackhole-2ch"])
        .status()
        .context("Failed to run Homebrew. Is it installed?")?;
    if !status.success() {
        eprintln!("[WARN] BlackHole installation failed.");
        return Ok(None);
    }

    println!("[INSTALL] BlackHole installer should restart the audio server automatically.");
    println!("[INSTALL] If the device doesn't appear, try: sudo launchctl kickstart -p system/com.apple.audio.coreaudiod");
    std::thread::sleep(std::time::Duration::from_secs(3));

    // refresh device list
    let host = cpal::default_host();
    for (i, d) in host.output_devices()?.enumerate() {
        if let Ok(desc) = d.description() {
            if desc.name().to_lowercase().contains("blackhole") {
                println!("[INSTALL] BlackHole 2ch installed successfully.");
                return Ok(Some(i));
            }
        }
    }
    eprintln!("[WARN] BlackHole installed but not found in device list. Try restarting.");
    Ok(None)
}

fn prompt_set_default_input(device_name: &str) -> Result<()> {
    print!("\nSet \"{}\" as the default input device on macOS? [Y/n]: ", device_name);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    if input.trim().to_lowercase() != "n" {
        set_default_input_device(device_name)?;
    }
    Ok(())
}

fn set_default_input_device(device_name: &str) -> Result<()> {
    unsafe {
        let address = AudioObjectPropertyAddress {
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain,
        };

        // get device list size
        let mut data_size: u32 = 0;
        let status = AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            &address,
            0,
            std::ptr::null(),
            &mut data_size,
        );
        if status != kAudioHardwareNoError as i32 {
            anyhow::bail!("Failed to get device list size");
        }

        let device_count = data_size as usize / mem::size_of::<AudioDeviceID>();
        let mut device_ids = vec![0u32; device_count];
        let mut data_size = data_size;

        let status = AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &address,
            0,
            std::ptr::null(),
            &mut data_size,
            device_ids.as_mut_ptr() as *mut std::ffi::c_void,
        );
        if status != kAudioHardwareNoError as i32 {
            anyhow::bail!("Failed to get device list");
        }

        // find the target device by name
        let mut target_id = None;
        for &device_id in &device_ids {
            let name_addr = AudioObjectPropertyAddress {
                mSelector: kAudioDevicePropertyDeviceName,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain,
            };
            let mut name_buf = [0u8; 256];
            let mut name_size: u32 = 256;
            let status = AudioObjectGetPropertyData(
                device_id,
                &name_addr,
                0,
                std::ptr::null(),
                &mut name_size,
                name_buf.as_mut_ptr() as *mut std::ffi::c_void,
            );
            if status == kAudioHardwareNoError as i32 {
                if let Ok(c_str) = CStr::from_ptr(name_buf.as_ptr() as *const std::os::raw::c_char).to_str() {
                    if c_str.to_lowercase().contains(&device_name.to_lowercase()) {
                        target_id = Some(device_id);
                        break;
                    }
                }
            }
        }

        let device_id = match target_id {
            Some(id) => id,
            None => anyhow::bail!("Device \"{}\" not found in CoreAudio device list", device_name),
        };

        // set as default input device
        let set_addr = AudioObjectPropertyAddress {
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain,
        };
        let status = AudioObjectSetPropertyData(
            kAudioObjectSystemObject,
            &set_addr,
            0,
            std::ptr::null(),
            mem::size_of::<AudioDeviceID>() as u32,
            &device_id as *const AudioDeviceID as *const std::ffi::c_void,
        );
        if status != kAudioHardwareNoError as i32 {
            anyhow::bail!("Failed to set default input device (error: {})", status);
        }
        println!("[OK] \"{}\" set as default input device.", device_name);
        Ok(())
    }
}

fn play_tts(text: &str, state: Arc<Mutex<MixerState>>) -> Result<()> {
    let sample_rate = state.lock().unwrap().sample_rate;
    let history_dir = "tts_history";
    std::fs::create_dir_all(history_dir)?;
    let tmp_file = format!("{}/tts_{}.wav", history_dir, chrono::Local::now().format("%Y%m%d_%H%M%S"));
    
    let status = Command::new("say")
        .args(&["-o", &tmp_file, "--file-format=WAVE", &format!("--data-format=LEF32@{}", sample_rate), text])
        .status()?;
    if !status.success() { anyhow::bail!("say failed"); }
    
    let mut reader = hound::WavReader::open(&tmp_file)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect();
    let buffer = if spec.channels == 1 {
        samples.into_iter().flat_map(|s| std::iter::repeat(s).take(CHANNELS as usize)).collect()
    } else {
        samples
    };

    let mut s = state.lock().unwrap();
    s.tts_buffer = buffer;
    s.tts_main_idx = 0;
    s.tts_mon_idx = 0;
    println!("[Mixer] Injecting TTS on mic: \"{}\"", text);
    Ok(())
}

fn start_monitor(host: &cpal::Host, state: &Arc<Mutex<MixerState>>) -> Option<cpal::Stream> {
    let mon_device = host.default_output_device()?;
    let mon_config = match mon_device.default_output_config() {
        Ok(c) => c.config(),
        Err(e) => { eprintln!("[Error] Monitor config failed: {}", e); return None; }
    };
    let mon_state = Arc::clone(state);
    let mode_label = {
        let s = state.lock().unwrap();
        if s.tts_only_monitor { "TTS only" } else { "Mic + TTS" }
    };
    match mon_device.build_output_stream(
        &mon_config,
        move |data, _| mon_state.lock().unwrap().fill_mon(data),
        |err| eprintln!("[Error] Monitor: {}", err),
        None
    ) {
        Ok(stream) => {
            if let Err(e) = stream.play() {
                eprintln!("[Error] Could not play monitor stream: {}", e);
                None
            } else {
                let dev_name = mon_device.description().map(|d| d.name().to_owned()).unwrap_or_else(|_| "Unknown".into());
                println!("[MONITOR] Activated on {} ({})", dev_name, mode_label);
                Some(stream)
            }
        }
        Err(e) => {
            eprintln!("[Error] Could not build monitor stream: {}", e);
            None
        }
    }
}
