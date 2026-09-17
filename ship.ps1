param(
    [string]$Message = "",
    [string]$Remote = "origin",
    [string]$Branch = ""
)
# One-command commit + push.
#   .\ship.ps1                    # auto message, push current branch to origin
#   .\ship.ps1 "Fix greeting"     # your own message
#   .\ship.ps1 "" ai-girlfriend   # push to another remote
$ErrorActionPreference = "Stop"

git rev-parse --show-toplevel | Out-Null

if ([string]::IsNullOrWhiteSpace($Branch)) {
    $Branch = (git branch --show-current).Trim()
}

git add -A
$staged = @(git diff --cached --name-only | Where-Object { $_ -ne "" })
if ($staged.Count -eq 0) {
    Write-Output "nothing to commit, working tree clean"
    exit 0
}

# Safety: never push private chat logs or huge weight files (GitHub caps at 100MB)
if ($staged -contains "data/mychat.txt") {
    Write-Error "BLOCKED: data/mychat.txt (private chat log) is staged. Untrack it: git rm --cached data/mychat.txt"
}
$big = @($staged | Where-Object {
    (Test-Path -LiteralPath $_ -PathType Leaf) -and ((Get-Item -LiteralPath $_).Length -gt 90MB)
})
if ($big.Count -gt 0) {
    Write-Error ("BLOCKED files over 90MB (GitHub rejects 100MB+): " + ($big -join ", "))
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $names = ($staged | Split-Path -Leaf) | Select-Object -Unique | Select-Object -First 3
    $more = if ($staged.Count -gt 3) { " (+$($staged.Count - 3) more)" } else { "" }
    $Message = "Update $($names -join ', ')$more — $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}

git commit -m $Message

# Set upstream on first push so plain `git push` works afterwards
$upstream = git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null
if ([string]::IsNullOrWhiteSpace($upstream)) {
    git push -u $Remote $Branch
} else {
    git push $Remote $Branch
}
