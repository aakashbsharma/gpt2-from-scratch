/**
 * The model registry, loaded once and cached for the session.
 *
 * `ComparePanel` needs two *ready* models; a model whose checkpoint is missing is
 * still listed by the backend with `ready: false`, so filtering has to happen
 * here rather than by assuming the list is usable.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import { fetchModels, invalidateModelsCache } from '@/lib/api';
import { STATIC_ONLY } from '@/lib/env';
import type { ModelInfo } from '@/types/trace';

export interface ModelRegistry {
  models: ModelInfo[];
  /** Models that are loaded and can actually run a forward pass. */
  ready: ModelInfo[];
  loading: boolean;
  error: string | null;
  /** True when the registry could not be reached, i.e. no backend is running. */
  offline: boolean;
  refresh: () => void;
}

export function useModels(): ModelRegistry {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    // A static bundle has no registry to ask. `STATIC_ONLY` is a compile-time
    // constant, so this whole effect is dropped from the static build.
    if (STATIC_ONLY) {
      setLoading(false);
      setOffline(true);
      setError('static build: no backend, so the model registry is unavailable');
      return;
    }

    // A local flag rather than an AbortController: the underlying request is
    // shared and cached, so one consumer unmounting must not cancel it out from
    // under the others. This just decides whether *this* consumer still cares.
    let cancelled = false;
    setLoading(true);

    fetchModels()
      .then((list) => {
        if (cancelled) return;
        setModels(list);
        setError(list.length ? null : 'no models reported by the backend');
        setOffline(list.length === 0);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [nonce]);

  const refresh = useCallback(() => {
    invalidateModelsCache();
    setNonce((n) => n + 1);
  }, []);

  // `models.filter` allocates a new array each render. This is a derived value,
  // not a zustand selector, so it is safe here — `ready` never reaches
  // `useSyncExternalStore`, only React's own render path.
  const ready = useMemo(() => models.filter((m) => m.ready), [models]);

  return { models, ready, loading, error, offline, refresh };
}
