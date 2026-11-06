# LSD-XAI frontend

Local research interface built with React, TypeScript, Vite, Tailwind CSS v4, and Recharts.

From this directory, use `npm run dev` to serve the app at http://localhost:5173. The API defaults to http://localhost:8000. To override it, copy `.env.example` to `.env.local` and set `VITE_API_URL` before starting Vite.

- `npm run build`: TypeScript checks and production build.
- `npm run lint`: Oxlint checks.
- `npm run preview`: preview the production build.

The Diagnose page uses the API's default binary or three-class run. Results highlights the run with the highest reported F1; comparisons should account for the classification task. No image or model data is bundled with the frontend.
