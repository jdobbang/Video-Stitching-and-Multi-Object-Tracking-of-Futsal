# YOLOv8n ONNX 다운로드 스크립트

Write-Host "YOLOv8n ONNX 모델 다운로드 중..." -ForegroundColor Yellow
Write-Host ""

# 다운로드 URL (여러 소스 시도)
$urls = @(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
    "https://media.githubusercontent.com/media/ultralytics/assets/main/yolov8n.onnx"
)

$output = "yolov8n.onnx"
$success = $false

foreach ($url in $urls) {
    Write-Host "시도 중: $url" -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $url -OutFile $output -ErrorAction Stop
        $fileSize = (Get-Item $output).Length / 1MB
        Write-Host "✅ 다운로드 성공!" -ForegroundColor Green
        Write-Host "파일: $output ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Green
        $success = $true
        break
    }
    catch {
        Write-Host "❌ 실패: $_" -ForegroundColor Red
        Write-Host ""
    }
}

if (-not $success) {
    Write-Host ""
    Write-Host "모든 자동 다운로드 실패" -ForegroundColor Red
    Write-Host ""
    Write-Host "수동 다운로드 방법:" -ForegroundColor Yellow
    Write-Host "1. 브라우저에서 열기: https://github.com/ultralytics/ultralytics" -ForegroundColor White
    Write-Host "2. YOLOv8 문서에서 ONNX 모델 다운로드" -ForegroundColor White
    Write-Host "3. 다운로드한 파일을 이 폴더에 yolov8n.onnx로 저장" -ForegroundColor White
    Write-Host ""
    Write-Host "현재 폴더: $PWD" -ForegroundColor Cyan
}
