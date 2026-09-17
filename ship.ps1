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
    # Content-aware message: per-file +added/-deleted lines plus added def/class names
    $addedFiles = @(git diff --cached --diff-filter=A --name-only | Where-Object { $_ -ne "" })
    $deletedFiles = @(git diff --cached --diff-filter=D --name-only | Where-Object { $_ -ne "" })
    $numstat = @(git diff --cached --numstat | Where-Object { $_ -ne "" })
    $syms = @(git diff --cached -U0 -- '*.py' |
        Where-Object { $_ -match '^\+(def |class )' } |
        ForEach-Object { ($_ -replace '^\+(def |class )', '') -replace '[(:].*', '' } |
        Select-Object -Unique | Select-Object -First 4)
    $bits = @()
    foreach ($f in ($addedFiles | Split-Path -Leaf)) { $bits += "new $f" }
    foreach ($line in $numstat) {
        $a, $d, $f = ($line -split "`t")
        if (($addedFiles -contains $f) -or ($deletedFiles -contains $f)) { continue }
        $bits += "$(Split-Path $f -Leaf) +$a/-$d"
    }
    foreach ($f in ($deletedFiles | Split-Path -Leaf)) { $bits += "drop $f" }
    $Message = $bits -join "; "
    if ($syms.Count -gt 0) { $Message += " ($($syms -join ', '))" }
    if ($Message.Length -gt 160) { $Message = $Message.Substring(0, 157) + "..." }
}

git commit -m $Message

# Set upstream on first push so plain `git push` works afterwards
$upstream = git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null
if ([string]::IsNullOrWhiteSpace($upstream)) {
    git push -u $Remote $Branch
} else {
    git push $Remote $Branch
}
