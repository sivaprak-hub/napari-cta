# napari-cta — Version Timeline

| # | Commit  | Date       | Files Changed               | What Changed |
|---|---------|------------|-----------------------------|--------------|
| 1 | c4e3ae8 | 2026-06-16 | README.md                   | First commit — project documentation |
| 2 | 1cc4a59 | 2026-06-16 | backend.py, widget.py, config | All source files added (v0.1 baseline) |
| 3 | b50acca | 2026-06-16 | .gitignore                  | Removed egg-info from tracking |
| 4 | b8f8168 | 2026-06-29 | pyproject.toml              | Fixed package discovery config |
| 5 | 3dc2018 | 2026-07-09 | backend.py (+290), widget.py (+182) | Major: FPS detection, kinetics rewrite, exp decay fit, per-beat averaging |
| 6 | bfe8bce | 2026-07-19 | widget.py (+62)             | Per-file UI state persistence (restore layers/scores/chart on file switch) |
| 7 | dfe0e71 | 2026-07-24 | backend.py (+55), widget.py (+622) | Raw signal F0 fix, decay window split, full dark theme stylesheet |
| 8 | 9103784 | 2026-07-24 | backend.py (+19), widget.py (+879) | UI polish, graph window, layer controls, CD formula fix |
| 9 | 37ecd5f | 2026-07-29 | backend.py (+68), widget.py (+153) | Bug fixes: BPM, peak markers, QSpinBox, per-cell kinetics |
| 10| f20f729 | 2026-07-31 | widget.py (+213)            | Full light/dark theme — CTA responds to napari theme switches |
