param(
    [string]$WorkDir = "D:\Kairos-Track4-work",
    [string]$TeamId = "Kairos",
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $PythonPath) {
    $PythonPath = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment not found: $PythonPath"
}
if (-not $TeamId.Trim()) {
    throw "TeamId must not be empty."
}

$WorkDir = [IO.Path]::GetFullPath($WorkDir)
$WeightsDir = Join-Path $WorkDir "weights"
$TempDir = Join-Path $WorkDir "temp"
$PipCache = Join-Path $WorkDir "pip-cache"
$HFHome = Join-Path $WorkDir "hf-cache"
New-Item -ItemType Directory -Force -Path $WeightsDir, $TempDir, $PipCache, $HFHome | Out-Null

$workDrive = [IO.Path]::GetPathRoot($WorkDir).TrimEnd('\').TrimEnd(':')
$drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$($workDrive):'"
if (-not $drive -or $drive.FreeSpace -lt 20GB) {
    throw "The work drive must have at least 20 GB free before setup."
}
$systemDrive = [IO.Path]::GetPathRoot($Root).TrimEnd('\')
$systemDisk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$systemDrive'"
if (-not $systemDisk -or $systemDisk.FreeSpace -lt 8GB) {
    throw "The Python environment drive must have at least 8 GB free before installing CUDA PyTorch."
}

$ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegCommand) {
    $packageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    $ffmpegBin = Get-ChildItem -LiteralPath $packageRoot -Directory -Filter "Gyan.FFmpeg_*" -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -Filter "ffmpeg-*-full_build" } |
        ForEach-Object { Join-Path $_.FullName "bin" } |
        Where-Object { Test-Path (Join-Path $_ "ffmpeg.exe") } |
        Select-Object -First 1
    if ($ffmpegBin) {
        $env:Path = "$ffmpegBin;$env:Path"
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$ffmpegBin*") {
            [Environment]::SetEnvironmentVariable(
                "Path",
                "$ffmpegBin;$userPath",
                "User"
            )
        }
    }
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or
    -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "ffmpeg and ffprobe must be installed and available on PATH."
}

$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:PIP_CACHE_DIR = $PipCache
$env:HF_HOME = $HFHome
$env:KAIROS_WEIGHTS_DIR = $WeightsDir
$env:KAIROS_AVION_ROOT = Join-Path $Root "third_party\AVION"
$env:TEAM_ID_STR = $TeamId
$env:PYTHONPATH = Join-Path $Root "code"
[Environment]::SetEnvironmentVariable("KAIROS_WEIGHTS_DIR", $WeightsDir, "User")
[Environment]::SetEnvironmentVariable("KAIROS_AVION_ROOT", $env:KAIROS_AVION_ROOT, "User")
[Environment]::SetEnvironmentVariable("HF_HOME", $HFHome, "User")
[Environment]::SetEnvironmentVariable("TEAM_ID_STR", $TeamId, "User")

Write-Output "Installing CUDA PyTorch for the NVIDIA GPU..."
& $PythonPath -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }

$systemDisk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$systemDrive'"
if ($systemDisk.FreeSpace -lt 4GB) {
    throw "Less than 4 GB remains on the Python environment drive; stopping before optional packages."
}

Write-Output "Installing the project's model and video-download dependencies..."
& $PythonPath -m pip install --no-cache-dir -r (Join-Path $Root "requirements-models.txt")
if ($LASTEXITCODE -ne 0) { throw "Model dependency installation failed." }
& $PythonPath -m pip install --no-cache-dir yt-dlp
if ($LASTEXITCODE -ne 0) { throw "yt-dlp installation failed." }

$priorSource = Join-Path $Root "weights\epic_priors.npz"
$priorTarget = Join-Path $WeightsDir "epic_priors.npz"
Copy-Item -LiteralPath $priorSource -Destination $priorTarget -Force

$epicRepo = Join-Path $WorkDir "epic-kitchens-download-scripts"
if (-not (Test-Path (Join-Path $epicRepo "epic_downloader.py"))) {
    git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git $epicRepo
    if ($LASTEXITCODE -ne 0) { throw "Official EPIC downloader checkout failed." }
}

$avionDir = Join-Path $Root "third_party\AVION"
if (-not (Test-Path (Join-Path $avionDir ".git"))) {
    git clone --depth 1 https://github.com/zhaoyue-zephyrus/AVION.git $avionDir
    if ($LASTEXITCODE -ne 0) { throw "AVION source checkout failed." }
}
& $PythonPath (Join-Path $Root "tools\prepare_avion.py") $avionDir
if ($LASTEXITCODE -ne 0) { throw "AVION Windows compatibility preparation failed." }

Write-Output "Downloading SigLIP and VideoMAE checkpoints to $WeightsDir..."
& $PythonPath (Join-Path $Root "tools\download_models.py")
if ($LASTEXITCODE -ne 0) { throw "Hugging Face model download failed." }

Write-Output "Downloading and verifying the AVION checkpoint..."
& $PythonPath (Join-Path $Root "tools\download_avion.py")
if ($LASTEXITCODE -ne 0) { throw "AVION checkpoint download or verification failed." }

& $PythonPath -c "import torch,sys; print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch cannot access the NVIDIA GPU." }
& $PythonPath (Join-Path $Root "tools\environment_check.py")
if ($LASTEXITCODE -ne 0) { throw "Environment check failed." }
& $PythonPath -c "import os,numpy as np; from pathlib import Path; from router_siglip import SigLIPRouter; model=SigLIPRouter(Path(os.environ['KAIROS_WEIGHTS_DIR'])); label,score,probs,features=model.route_and_label(np.zeros((1,224,224,3),dtype=np.uint8)); print('SigLIP smoke:',label,features.shape); raise SystemExit(0 if model.enabled and features.shape[0]==1 else 1)"
if ($LASTEXITCODE -ne 0) { throw "SigLIP checkpoint failed to load." }
& $PythonPath -c "import numpy as np; from videomae_fpv import VideoMAEFpvClassifier; model=VideoMAEFpvClassifier(); scores=model.predict(np.zeros((16,224,224,3),dtype=np.uint8)); ok=model.enabled and np.isfinite(list(scores.values())).all(); print('VideoMAE smoke:',max(scores,key=scores.get)); raise SystemExit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) { throw "VideoMAE checkpoint failed to load or run." }
& $PythonPath -c "import os,sys,numpy as np; from pathlib import Path; from avion_epic import AVIONEpicClassifier; weights=Path(os.environ['KAIROS_WEIGHTS_DIR']); model=AVIONEpicClassifier(weights/'avion'/'avion_finetune_cls_lavila_vitb_best.pt',Path(os.environ['KAIROS_AVION_ROOT'])); mapped=model.enabled and model.load_action_mapping(weights/'epic_priors.npz'); logits=model.predict_joint_logits(np.zeros((1,16,224,224,3),dtype=np.uint8),batch_size=1) if mapped else np.empty((0,3806)); ok=mapped and logits.shape==(1,3806) and np.isfinite(logits).all(); print('AVION forward:',logits.shape); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) { throw "AVION checkpoint or EPIC action mapping failed to load." }

Write-Output ""
Write-Output "GPU setup complete. Weights and caches use: $WeightsDir"
Write-Output "Next: run tools\stream_manifest.py with --work-dir '$WorkDir' to process one source at a time."
