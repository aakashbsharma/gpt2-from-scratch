/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Empty in dev, where Vite proxies `/api`; an origin in a static deploy. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
