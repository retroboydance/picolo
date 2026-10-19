# Windows Work Laptop Environment Setup (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=== LLRE Setup: Windows (CPU Mode) ===" -ForegroundColor Cyan

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Green
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
}

Write-Host "Installing LLRE dependencies..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[smt,dev]"

Write-Host "Verifying LLRE runtime configuration..."
& .\.venv\Scripts\python.exe -m llre.config
