$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot 'openalex'
New-Item -ItemType Directory -Path $root -Force | Out-Null
$requests = @(
    @{name='safe_linear_bandits'; url='https://api.openalex.org/works/doi:10.1609/aaai.v34i06.6581'},
    @{name='uncertainty_sets'; url='https://api.openalex.org/works/doi:10.48550/arXiv.2004.03069'},
    @{name='nonstationary_chance_constraints'; url='https://api.openalex.org/works/doi:10.48550/arXiv.2205.03748'},
    @{name='wasserstein_regret_control'; url='https://api.openalex.org/works/doi:10.48550/arXiv.2304.06783'},
    @{name='wasserstein_dro'; url='https://api.openalex.org/works/doi:10.1007/s10107-017-1172-1'},
    @{name='conformal_covariate_shift'; url='https://api.openalex.org/works/doi:10.48550/arXiv.1904.06019'},
    @{name='safe_control_gym'; url='https://api.openalex.org/works/doi:10.1109/LRA.2022.3196132'},
    @{name='botorch'; url='https://api.openalex.org/works/doi:10.48550/arXiv.1910.06403'},
    @{name='moment_regret_control'; url='https://api.openalex.org/works/doi:10.48550/arXiv.2512.10906'},
    @{name='regret_minimization'; url='https://api.openalex.org/works/doi:10.48550/arXiv.2412.15406'},
    @{name='materials_bax'; url='https://api.openalex.org/works/doi:10.1038/s41524-024-01326-2'},
    @{name='safeopt_title_search'; url='https://api.openalex.org/works?filter=title.search:Safe%20Exploration%20for%20Optimization%20with%20Gaussian%20Processes&per-page=10'}
)
foreach ($request in $requests) {
    try {
        $response = Invoke-RestMethod -Uri $request.url
        if ($response -is [string]) { $response = $response | ConvertFrom-Json -AsHashtable }
        $record = [ordered]@{retrieved_at=(Get-Date).ToUniversalTime().ToString('o'); url=$request.url; response=$response}
        $record | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $root ($request.name+'.json')) -Encoding utf8
        if ($null -ne $response.results) {
            $selected = $response.results | ForEach-Object { [ordered]@{id=$_.id; doi=$_.doi; title=$_.title; publication_year=$_.publication_year} }
            [ordered]@{name=$request.name; count=$response.meta.count; results=$selected} | ConvertTo-Json -Depth 5
        } else {
            [ordered]@{name=$request.name; id=$response.id; title=$response.title; doi=$response.doi} | ConvertTo-Json -Depth 5
        }
    } catch {
        [ordered]@{name=$request.name; url=$request.url; error=$_.Exception.Message} | ConvertTo-Json
    }
}
