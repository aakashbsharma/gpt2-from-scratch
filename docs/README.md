# Documentation and media

| File | What it is |
| --- | --- |
| `architecture.md` | model structure, tracing, the API layer, measured tolerances |
| `trace-schema.md` | the JSON contract — source of truth for `backend/app/schemas.py` and `frontend/src/types/trace.ts` |
| `demo.gif` | **placeholder** — see below |

## Recording `demo.gif`

The README embeds `demo.gif`. It is not committed, so the image is currently
broken until you supply one.

Suggested sequence (keeps it under ~15 s at a readable pace):

1. `docker compose up --build`, open http://localhost:8080
2. Replay mode → `effort`. Press **play**.
3. While the token strip fills, click **Block 6** on the pipeline.
4. Switch Attention from `heatmap` to `arcs`, pick **head 9**.
5. Click the last token in the strip to move the query position.
6. Let it run back to the end so the LogitLens rows settle on a final prediction.

Then resize — a 15 s GIF at 12 fps and 800 px wide lands around 2–4 MB:

```bash
# macOS / Linux, from a .mov screen recording
ffmpeg -i capture.mov -vf "fps=12,scale=800:-1:flags=lanczos" -loop 0 demo.gif

# Windows (via WSL or the same binary)
ffmpeg -i capture.mp4 -vf "fps=12,scale=800:-1:flags=lanczos" -loop 0 demo.gif
```

For a terminal recording instead, [asciinema](https://asciinema.org) plus
`svg-term` produces a much smaller asset and suits the Python-side demo:

```bash
python -m scripts.export_traces
python -m scripts.generate_stream_demo  # or a short script using generate_stream
asciinema rec demo.cast
svg-term -c demo.cast -o demo.svg
```

Committing a ~3 MB binary is a deliberate cost for a README that explains
itself visually; keeping it out of version control until then is not an
accident, so the table above records the gap rather than hiding it.
