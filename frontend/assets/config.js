// Deployment configuration for the static frontend.
//
// API_BASE_URL is intentionally empty, which means "call the same origin that
// served this page" (the normal local-development and Render setup).
//
// When the frontend is hosted on Vercel and the API on Render, set this to the
// Render origin, e.g. "https://your-app.onrender.com", and add that Vercel
// origin to the backend's CORS_ORIGINS variable.
window.TRIPMATE_CONFIG = {
  API_BASE_URL: ""
};
