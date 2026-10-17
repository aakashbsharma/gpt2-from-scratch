# Build the static bundle, then serve it with nginx.
#
# Two stages so the runtime image carries no node_modules and no build toolchain.
# The nginx config also proxies `/api` to the backend, which keeps the browser on
# one origin: no CORS preflight, and the WebSocket upgrade works without any
# client-side origin configuration.

FROM node:20-alpine AS build
WORKDIR /app

# Copy manifests first so `npm ci` is cached until dependencies actually change.
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json vite.config.ts index.html postcss.config.js tailwind.config.js ./
COPY src ./src
COPY public ./public

# STATIC=1 builds the replay-only bundle: no live mode, no backend calls. The
# traces in public/ must be present, and `build:static` fails with an actionable
# message if they are not.
ARG STATIC=""
RUN if [ -n "$STATIC" ]; then npm run build:static; else npm run build; fi

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
