# Push NVIDIA GPU toward maximum sustained performance before long training.
# Run in an elevated PowerShell if persistence mode fails without admin.

param(
    [int]$GpuIndex = 0
)

Write-Host "=== CamRec GPU max performance (GPU $GpuIndex) ==="

# Persistence mode keeps driver ready (helps multi-run sessions).
nvidia-smi -i $GpuIndex -pm 1 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[warn] persistence mode failed (may need Admin). Continuing..."
}

# Power limit to the board maximum (many GeForce boards already run at max).
$maxPower = (nvidia-smi -i $GpuIndex --query-gpu=power.max_limit --format=csv,noheader,nounits).Trim()
if ($maxPower) {
    nvidia-smi -i $GpuIndex -pl $maxPower
    Write-Host "[ok] power limit -> $maxPower W"
}

# Prefer maximum performance mode where supported (Windows often ignores this).
nvidia-smi -i $GpuIndex -lgc 0,3090 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[info] lock graphics clocks not supported on this driver/SKU (normal on GeForce)."
}

Write-Host ""
nvidia-smi -i $GpuIndex --query-gpu=name,persistence_mode,power.limit,power.max_limit,clocks.max.sm,clocks.max.memory,memory.total,memory.free --format=csv
Write-Host ""
Write-Host "Tip: Windows power plan -> High performance / Ultimate performance"
Write-Host "Tip: NVIDIA Control Panel -> Manage 3D settings -> Power management mode = Prefer maximum performance"
Write-Host "Tip: Close browsers/games to free VRAM before AutoBatch"
