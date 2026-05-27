use anyhow::{Context, Result};
use coreaudio_sys::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::ffi::CStr;
use std::mem;
use std::process::Command;
use std::sync::{Arc, Mutex};

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const CHANNELS: u16 = 2;
const RING_BUFFER_MS: usize = 1000;

const LOOPBACK_KEYWORDS: &[&str] = &[
    "blackhole", "loopback", "soundflower",
    "eqmac", "menubus", "virtual", "wavtap",
];

struct MixerState {
    mic_ring: Vec<f32>,
    mic_write: usize,
    main_read: usize,
    main_count: usize,
    main_started: bool,
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
    mic_volume: f32,
    tts_volume: f32,
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
            mic_volume: 1.0,
            tts_volume: 1.0,
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
                val = self.mic_ring[self.main_read] * self.mic_volume;
                self.main_read = (self.main_read + 1) % self.mic_ring.len();
                self.main_count -= 1;
            }
            if self.tts_main_idx < self.tts_buffer.len() {
                val += self.tts_buffer[self.tts_main_idx] * self.tts_volume;
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
                val = self.mic_ring[self.mon_read] * self.mic_volume;
                self.mon_read = (self.mon_read + 1) % self.mic_ring.len();
                self.mon_count -= 1;
            }
            if self.tts_mon_idx < self.tts_buffer.len() {
                val += self.tts_buffer[self.tts_mon_idx] * self.tts_volume;
                self.tts_mon_idx += 1;
            }
            *sample = val;
        }
    }

    fn read_signal_levels(&mut self) -> (f32, f32) {
        let mic = self.mic_signal;
        let out = self.out_signal;
        self.mic_signal = 0.0;
        self.out_signal = 0.0;
        (mic, out)
    }

    fn is_tts_active(&self) -> bool {
        self.tts_main_idx < self.tts_buffer.len()
    }
}

struct AppState {
    mixer: Arc<Mutex<Option<MixerState>>>,
    output_stream: Arc<Mutex<Option<cpal::Stream>>>,
    input_stream: Arc<Mutex<Option<cpal::Stream>>>,
    monitor_stream: Arc<Mutex<Option<cpal::Stream>>>,
    original_default_input: Arc<Mutex<Option<String>>>,
}

#[derive(Serialize)]
struct StatusResponse {
    audio_running: bool,
    mic_signal: f32,
    out_signal: f32,
    mic_volume: f32,
    tts_volume: f32,
    monitor_mode: String,
    tts_active: bool,
}

#[derive(Deserialize)]
struct TtsRequest { text: String }

#[derive(Deserialize)]
struct VolumeRequest {
    mic: Option<f32>,
    tts: Option<f32>,
}

#[derive(Deserialize)]
struct MonitorModeRequest { mode: String }

#[derive(Deserialize)]
struct StartRequest {
    input: usize,
    output: usize,
    set_default_input: Option<bool>,
}

// --- Web handlers ---

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../web/index.html"))
}

fn build_status(state: &AppState) -> StatusResponse {
    let mut guard = state.mixer.lock().unwrap();
    if let Some(ref mut mixer) = *guard {
        let (mic_signal, out_signal) = mixer.read_signal_levels();
        let has_monitor = state.monitor_stream.lock().unwrap().is_some();
        let monitor_mode = if !has_monitor { "off" }
            else if mixer.tts_only_monitor { "tts-only" }
            else { "monitor" };
        StatusResponse {
            audio_running: true,
            mic_signal,
            out_signal,
            mic_volume: mixer.mic_volume,
            tts_volume: mixer.tts_volume,
            monitor_mode: monitor_mode.to_string(),
            tts_active: mixer.is_tts_active(),
        }
    } else {
        StatusResponse {
            audio_running: false,
            mic_signal: 0.0,
            out_signal: 0.0,
            mic_volume: 1.0,
            tts_volume: 1.0,
            monitor_mode: "off".to_string(),
            tts_active: false,
        }
    }
}

async fn status_handler(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    Json(build_status(&state))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(150));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let status = build_status(&state);
                if let Ok(json) = serde_json::to_string(&status) {
                    if socket.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Text(t))) => {
                        if t == "ping" {
                            let _ = socket.send(Message::Text("pong".into())).await;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

async fn tts_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TtsRequest>,
) -> impl IntoResponse {
    let text = body.text;
    if text.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "Text is empty"})));
    }

    let has_mixer = state.mixer.lock().unwrap().is_some();
    if !has_mixer {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "Audio not running"})));
    }

    let mixer = state.mixer.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = play_tts(&text, mixer) {
            eprintln!("[Error] TTS: {}", e);
        }
    });

    (StatusCode::OK, Json(json!({"success": true})))
}

async fn monitor_mode_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<MonitorModeRequest>,
) -> Json<serde_json::Value> {
    let mode = body.mode.to_lowercase();
    let mut monitor = state.monitor_stream.lock().unwrap();
    let mut mixer_guard = state.mixer.lock().unwrap();
    let mixer = match mixer_guard.as_mut() {
        Some(m) => m,
        None => return Json(json!({"error": "Audio not running"})),
    };

    match mode.as_str() {
        "off" => {
            *monitor = None;
            println!("[MONITOR] Deactivated");
        }
        "monitor" => {
            mixer.tts_only_monitor = false;
            if monitor.is_none() {
                drop(mixer_guard);
                *monitor = start_monitor(&state.mixer);
            } else {
                println!("[MONITOR] Mode: Mic + TTS");
            }
        }
        "tts-only" => {
            mixer.tts_only_monitor = true;
            if monitor.is_none() {
                drop(mixer_guard);
                *monitor = start_monitor(&state.mixer);
            } else {
                println!("[MONITOR] Mode: TTS only");
            }
        }
        _ => return Json(json!({"error": "Invalid mode. Use: off, monitor, tts-only"})),
    }

    let current_mode = match monitor.is_some() {
        false => "off",
        true if mode == "tts-only" => "tts-only",
        true => "monitor",
    };
    Json(json!({"monitor_mode": current_mode}))
}

async fn volume_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<VolumeRequest>,
) -> Json<serde_json::Value> {
    let mut guard = state.mixer.lock().unwrap();
    let mixer = match guard.as_mut() {
        Some(m) => m,
        None => return Json(json!({"error": "Audio not running"})),
    };
    if let Some(v) = body.mic { mixer.mic_volume = v.clamp(0.0, 2.0); }
    if let Some(v) = body.tts { mixer.tts_volume = v.clamp(0.0, 2.0); }
    println!("[VOLUME] Mic: {:.2}, TTS: {:.2}", mixer.mic_volume, mixer.tts_volume);
    Json(json!({"mic_volume": mixer.mic_volume, "tts_volume": mixer.tts_volume}))
}

async fn devices_handler() -> Json<serde_json::Value> {
    let host = cpal::default_host();
    let input_devices: Vec<cpal::Device> = host.input_devices()
        .map(|d| d.collect())
        .unwrap_or_default();
    let output_devices: Vec<cpal::Device> = host.output_devices()
        .map(|d| d.collect())
        .unwrap_or_default();

    let default_input_device = host.default_input_device();

    let mut default_input = 0;
    let inputs: Vec<serde_json::Value> = input_devices.iter().enumerate().map(|(i, d)| {
        let name = d.description().map(|desc| desc.name().to_owned()).unwrap_or_else(|_| "Unknown".into());
        // Match the system's default input device by name
        if let Some(ref def) = default_input_device {
            if let Ok(def_desc) = def.description() {
                if def_desc.name().to_lowercase().trim() == name.to_lowercase().trim() {
                    default_input = i;
                }
            }
        }
        json!({"index": i, "name": name})
    }).collect();

    // Prefer BlackHole > any loopback > first device
    let mut default_output = None;
    let mut first_loopback = None;
    let outputs: Vec<serde_json::Value> = output_devices.iter().enumerate().map(|(i, d)| {
        let name = d.description().map(|desc| desc.name().to_owned()).unwrap_or_else(|_| "Unknown".into());
        let lower = name.to_lowercase();
        let is_loopback = LOOPBACK_KEYWORDS.iter().any(|k| lower.contains(k));
        if is_loopback {
            if first_loopback.is_none() { first_loopback = Some(i); }
            if lower.contains("blackhole") && default_output.is_none() { default_output = Some(i); }
        }
        json!({"index": i, "name": name, "is_loopback": is_loopback})
    }).collect();

    let default_output = default_output.or(first_loopback).unwrap_or(0);
    let has_loopback = first_loopback.is_some();
    let can_install = Command::new("brew").arg("--version").output().is_ok();

    Json(json!({
        "input_devices": inputs,
        "output_devices": outputs,
        "has_loopback": has_loopback,
        "can_install_blackhole": can_install,
        "default_input": default_input,
        "default_output": default_output,
    }))
}

async fn start_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<StartRequest>,
) -> impl IntoResponse {
    let host = cpal::default_host();

    let input_devices: Vec<cpal::Device> = match host.input_devices() {
        Ok(d) => d.collect(),
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("Failed to list input devices: {}", e)}))),
    };
    let output_devices: Vec<cpal::Device> = match host.output_devices() {
        Ok(d) => d.collect(),
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("Failed to list output devices: {}", e)}))),
    };

    let input_device = match input_devices.into_iter().nth(body.input) {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Input device not found"}))),
    };
    let output_device = match output_devices.into_iter().nth(body.output) {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Output device not found"}))),
    };

    let config = match output_device.default_output_config() {
        Ok(c) => c.config(),
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("Output config: {}", e)}))),
    };
    let sample_rate = config.sample_rate;
    let mut input_config = match input_device.default_input_config() {
        Ok(c) => c.config(),
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("Input config: {}", e)}))),
    };
    let mut input_channels = input_config.channels;
    println!("[CONFIG] Output: {} Hz, Input default: {} Hz ({} channels)", sample_rate, input_config.sample_rate, input_channels);

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

    // Warn if output is not a loopback device
    if let Ok(desc) = output_device.description() {
        let name = desc.name().to_owned();
        let is_loopback = LOOPBACK_KEYWORDS.iter().any(|k| name.to_lowercase().contains(k));
        if !is_loopback {
            eprintln!("[WARN] \"{}\" is not a loopback/virtual device.", name);
        }
    }

    // Stop any existing audio first
    stop_audio(&state);

    *state.mixer.lock().unwrap() = Some(MixerState::new(sample_rate, input_channels, false));

    let play_state = state.mixer.clone();
    let output_stream = match output_device.build_output_stream(
        &config,
        move |data: &mut [f32], _| {
            let mut guard = play_state.lock().unwrap();
            if let Some(ref mut m) = *guard { m.fill_main(data); }
        },
        |err| eprintln!("[Error] Output: {}", err),
        None,
    ) {
        Ok(s) => s,
        Err(e) => {
            *state.mixer.lock().unwrap() = None;
            return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("Output stream: {}", e)})));
        }
    };
    let _ = output_stream.play();

    let capture_state = state.mixer.clone();
    let input_stream = match input_device.build_input_stream(
        &input_config,
        move |data: &[f32], _| {
            let mut guard = capture_state.lock().unwrap();
            if let Some(ref mut m) = *guard { m.push_input(data); }
        },
        |err| eprintln!("[Error] Input: {}", err),
        None,
    ) {
        Ok(s) => s,
        Err(e) => {
            *state.mixer.lock().unwrap() = None;
            return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("Input stream: {}", e)})));
        }
    };
    let _ = input_stream.play();

    *state.output_stream.lock().unwrap() = Some(output_stream);
    *state.input_stream.lock().unwrap() = Some(input_stream);

    // Set as default macOS input device AFTER streams are running
    // to prevent macOS from reverting when it detects mic usage
    if body.set_default_input.unwrap_or(false) {
        if let Ok(desc) = output_device.description() {
            let name = desc.name().to_owned();
            let is_loopback = LOOPBACK_KEYWORDS.iter().any(|k| name.to_lowercase().contains(k));
            if is_loopback {
                let original = host.default_input_device()
                    .and_then(|d| d.description().ok())
                    .map(|desc| desc.name().to_owned());
                if let Some(ref orig) = original {
                    *state.original_default_input.lock().unwrap() = Some(orig.clone());
                }
                let _ = set_default_input_device(&name);
            }
        }
    }

    println!("[SYSTEM] Audio started");
    (StatusCode::OK, Json(json!({"success": true})))
}

async fn stop_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    stop_audio(&state);
    println!("[SYSTEM] Audio stopped");
    Json(json!({"success": true}))
}

async fn install_blackhole_handler() -> Json<serde_json::Value> {
    match install_blackhole_inner() {
        Ok(success) => Json(json!({"success": success})),
        Err(e) => Json(json!({"success": false, "error": format!("{}", e)})),
    }
}

fn stop_audio(state: &AppState) {
    *state.monitor_stream.lock().unwrap() = None;
    *state.output_stream.lock().unwrap() = None;
    *state.input_stream.lock().unwrap() = None;
    *state.mixer.lock().unwrap() = None;

    // Restore the original default input device if we changed it
    let original = state.original_default_input.lock().unwrap().take();
    if let Some(ref name) = original {
        println!("[RESTORE] Setting default input back to \"{}\"", name);
        let _ = set_default_input_device(name);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n=== TTS and Mic (Beta) ===");

    let app_state = Arc::new(AppState {
        mixer: Arc::new(Mutex::new(None)),
        output_stream: Arc::new(Mutex::new(None)),
        input_stream: Arc::new(Mutex::new(None)),
        monitor_stream: Arc::new(Mutex::new(None)),
        original_default_input: Arc::new(Mutex::new(None)),
    });

    let state_for_cleanup = Arc::clone(&app_state);

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/status", get(status_handler))
        .route("/ws", get(ws_handler))
        .route("/devices", get(devices_handler))
        .route("/start", post(start_handler))
        .route("/stop", post(stop_handler))
        .route("/tts", post(tts_handler))
        .route("/monitor-mode", post(monitor_mode_handler))
        .route("/volume", post(volume_handler))
        .route("/install-blackhole", post(install_blackhole_handler))
        .with_state(app_state);

    let addr = "127.0.0.1:17399";
    println!("[WEB] Interface: http://{}", addr);
    let _ = Command::new("open").arg(format!("http://{}", addr)).spawn();

    let listener = tokio::net::TcpListener::bind(addr).await?;

    let server = axum::serve(listener, app);
    let graceful = server.with_graceful_shutdown(async {
        tokio::signal::ctrl_c().await.ok();
        println!("\n[SHUTDOWN] Received Ctrl+C, cleaning up...");
    });

    graceful.await?;

    stop_audio(&state_for_cleanup);
    println!("[SHUTDOWN] Goodbye.");

    Ok(())
}

fn play_tts(text: &str, state: Arc<Mutex<Option<MixerState>>>) -> Result<()> {
    let sample_rate = {
        let guard = state.lock().unwrap();
        let mixer = guard.as_ref().context("Audio not running")?;
        mixer.sample_rate
    };
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

    let mut guard = state.lock().unwrap();
    let mixer = guard.as_mut().context("Audio not running")?;
    mixer.tts_buffer = buffer;
    mixer.tts_main_idx = 0;
    mixer.tts_mon_idx = 0;
    println!("[Mixer] Injecting TTS on mic: \"{}\"", text);
    Ok(())
}

fn start_monitor(state: &Arc<Mutex<Option<MixerState>>>) -> Option<cpal::Stream> {
    let host = cpal::default_host();
    let mon_device = host.default_output_device()?;
    let mon_state = Arc::clone(state);

    // Match monitor sample rate to the mixer's sample rate to prevent pitch shift
    let target_rate = {
        let guard = state.lock().unwrap();
        guard.as_ref()?.sample_rate
    };

    let mon_config = mon_device.supported_output_configs()
        .ok()?
        .find(|c| {
            c.sample_format() == cpal::SampleFormat::F32
                && c.min_sample_rate() <= target_rate
                && c.max_sample_rate() >= target_rate
        })
        .and_then(|c| c.try_with_sample_rate(target_rate))
        .map(|c| c.config())
        .or_else(|| {
            mon_device.default_output_config().ok().map(|c| c.config())
        })?;

    let mode_label = {
        let guard = state.lock().unwrap();
        let mixer = guard.as_ref()?;
        if mixer.tts_only_monitor { "TTS only" } else { "Mic + TTS" }
    };
    match mon_device.build_output_stream(
        &mon_config,
        move |data, _| {
            let mut guard = mon_state.lock().unwrap();
            if let Some(ref mut m) = *guard { m.fill_mon(data); }
        },
        |err| eprintln!("[Error] Monitor: {}", err),
        None,
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

fn install_blackhole_inner() -> Result<bool> {
    println!("[INSTALL] Installing BlackHole 2ch via Homebrew...");
    let status = Command::new("brew")
        .args(&["install", "--cask", "blackhole-2ch"])
        .status()
        .context("Failed to run Homebrew. Is it installed?")?;
    if !status.success() {
        eprintln!("[WARN] BlackHole installation failed.");
        return Ok(false);
    }
    println!("[INSTALL] BlackHole installed.");
    std::thread::sleep(std::time::Duration::from_secs(3));
    Ok(true)
}

fn set_default_input_device(device_name: &str) -> Result<()> {
    unsafe {
        let address = AudioObjectPropertyAddress {
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain,
        };
        let mut data_size: u32 = 0;
        let status = AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject, &address, 0, std::ptr::null(), &mut data_size,
        );
        if status != kAudioHardwareNoError as i32 {
            anyhow::bail!("Failed to get device list size");
        }
        let device_count = data_size as usize / mem::size_of::<AudioDeviceID>();
        let mut device_ids = vec![0u32; device_count];
        let mut data_size = data_size;
        let status = AudioObjectGetPropertyData(
            kAudioObjectSystemObject, &address, 0, std::ptr::null(),
            &mut data_size, device_ids.as_mut_ptr() as *mut std::ffi::c_void,
        );
        if status != kAudioHardwareNoError as i32 {
            anyhow::bail!("Failed to get device list");
        }
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
                device_id, &name_addr, 0, std::ptr::null(),
                &mut name_size, name_buf.as_mut_ptr() as *mut std::ffi::c_void,
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
            None => anyhow::bail!("Device \"{}\" not found", device_name),
        };
        let set_addr = AudioObjectPropertyAddress {
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain,
        };
        let status = AudioObjectSetPropertyData(
            kAudioObjectSystemObject, &set_addr, 0, std::ptr::null(),
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
