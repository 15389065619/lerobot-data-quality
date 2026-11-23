$ErrorActionPreference = 'Stop'

$dir = Join-Path $env:USERPROFILE '.kaggle'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

Write-Host '到 https://www.kaggle.com/settings/api 点 Generate New Token，复制那串 token' -ForegroundColor Cyan
Write-Host '粘贴时不会显示，属正常现象' -ForegroundColor Cyan
$secure = Read-Host 'API Token' -AsSecureString
$token = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
).Trim()

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host 'token 为空，退出' -ForegroundColor Red
    exit 1
}

$out = Join-Path $dir 'access_token'
[IO.File]::WriteAllText($out, $token, (New-Object Text.UTF8Encoding($false)))
Write-Host "已写入: $out" -ForegroundColor Green

# 旧的 username/key json 与新版 token 并存时会让人误判，这里清掉
$legacy = Join-Path $dir 'kaggle.json'
if (Test-Path $legacy) {
    Remove-Item $legacy -Force
    Write-Host '已移除旧的 kaggle.json（username/key 格式）' -ForegroundColor Yellow
}

Write-Host '正在验证...' -ForegroundColor Cyan
python -m kaggle datasets list -m 3
if ($LASTEXITCODE -eq 0) {
    Write-Host '凭据可用，回对话里说一声即可' -ForegroundColor Green
} else {
    Write-Host '验证失败：检查 token 是否过期或被截断' -ForegroundColor Yellow
}
