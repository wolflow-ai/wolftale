# Wolftale UI

React + Vite SPA for the Wolftale cloud demo.

## Local development

```bash
cd wolftale-ui
npm install

# Point at local API
echo "VITE_API_URL=http://localhost:8000" > .env.local

npm run dev
# → http://localhost:5173
```

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Wolftale API base URL | `http://localhost:8000` |

## Build

```bash
npm run build
# Output: dist/
```

## Deploying to Render

1. Push to GitHub (can be same repo as API, different directory)
2. Create a Static Site on Render, point at this directory
3. Build command: `npm install && npm run build`
4. Publish directory: `dist`
5. Set `VITE_API_URL` to your API's Render URL
6. Add custom domain: `tale.wolflow.ai`

The `render.yaml` in this directory handles this automatically if you use Render's Blueprint feature.

## Custom domain

After deploying, add `tale.wolflow.ai` as a custom domain in Render's dashboard.
Add a CNAME record pointing to your Render static site URL.

## Design notes

- `#0b0b0b` background, grain overlay — matches Wolfpath aesthetic
- DM Serif Display (headlines/chat) + DM Mono (system trace, labels) — two fonts, clear hierarchy
- `#c8f566` for memory/action elements, `#6b7fd4` for identity domain — your portfolio accent
- Three phases: idle (landing) → chat + memory panel (two-column)
- Memory panel shows claims forming in real time with domain color coding and confidence bars
- Gate decision trace visible under each assistant message (subtle, not intrusive)
- Full mobile responsiveness handled via CSS (memory panel collapses below chat on small screens)
