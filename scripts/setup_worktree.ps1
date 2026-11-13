param(
    [Parameter(Mandatory = $true)]
    [string]$BranchName,

    [string]$WorktreeRoot = "..\CamRec-worktrees"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

git rev-parse --verify HEAD *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Current repository has no commit yet. Create the first commit before using git worktree."
    exit 1
}

$targetRoot = Resolve-Path $WorktreeRoot -ErrorAction SilentlyContinue
if (-not $targetRoot) {
    $targetRoot = New-Item -ItemType Directory -Path $WorktreeRoot -Force
}

$targetPath = Join-Path $targetRoot $BranchName
git worktree add $targetPath -b $BranchName
