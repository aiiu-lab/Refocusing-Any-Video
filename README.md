# Refocusing Any Video Project Page

This directory contains the static project page for **Refocusing Any Video: Temporally Consistent Video Refocusing for Dynamic Focal Planes**.

The page is a self-contained HTML/CSS/JavaScript site that presents the paper abstract, qualitative results, full-frame baseline comparisons, zoom-inset comparisons, and quantitative tables for the SIGGRAPH Asia 2026 submission.

## Structure

```text
.
├── index.html              # Main project page
├── SA26_fast-forward.mp4   # SIGGRAPH Asia fast-forward video
├── static/
│   ├── css/index.css       # Page layout and responsive styling
│   └── js/index.js         # Navbar, video playback, and zoom carousel behavior
└── asset/
    ├── baselines/          # Six-scene baseline carousel videos and checksum manifest
    ├── results/            # Main result videos
    └── zoom_insets/        # Zoom-inset carousel videos
```

## Local Preview

No build step is required. Open `index.html` directly in a browser, or serve this directory with any static file server:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Asset Policy

The page uses `SA26_fast-forward.mp4` for the fast-forward presentation, browser-ready MP4 files in `asset/baselines/` for baseline comparisons, and compressed `*_web.mp4` files for the other results.

The 48 baseline videos (six scenes, eight videos per scene) were copied without re-encoding from `project_page_assets-20261020T064951Z-1-001/project_page_assets/`. They include the input, input and target focus maps, ground truth, DiffCamera, GenFocus, Learn2Refocus, and RAV. All use H.264/yuv420p with faststart; `asset/baselines/manifest.json` retains their sizes and SHA-256 checksums. The page references only the copies under `asset/`, so the original download folder and ZIP can be deleted.

When adding a new result:

1. Export a browser-friendly MP4, preferably named with the `*_web.mp4` suffix.
2. Place it under the matching folder in `asset/baselines/`, `asset/results/`, or `asset/zoom_insets/`.
3. Reference the file from `index.html`.
4. Run a quick broken-link check before publishing.

## Quantitative Results

The six tables in `index.html#quantitative` are transcribed from the final manuscript in `../SIGGRAPH_Asia_2026_RAV_Final/`:

- `sec/4_exp.tex`: video refocusing (`tab:main_quant`), Stage 1 (`tab:vdd_combined`), temporal sharpness (`tab:tsc_ablation`), and facial correspondence with Refiner (`tab:fc_ablation`).
- `sec/X_supp.tex`: full user-study response breakdown (`tab:supp_user_study_details`) and execution time (`supp:tab:runtime`).

Use the source table cells when nearby prose differs. Best/second-best styling follows numerical ranks. User-study percentages include neutral responses; runtime lists DiffCamera's different input resolution explicitly. HTML comments retain each table's source label for future updates.

## External Dependencies

The page loads these assets from CDNs:

- Google Fonts
- Bulma CSS
- Font Awesome CSS

The custom fallback styles in `static/css/index.css` cover the page's core layout for local viewing, but the CDN styles should remain available for the intended presentation.

## Maintenance Checklist

- Keep all video paths relative to `index.html`.
- Prefer compressed web MP4s over GIFs for page weight and playback performance.
- Avoid committing generated source videos unless they are directly used by the page.
- Test the page on both desktop and mobile widths after editing the layout or carousel.
