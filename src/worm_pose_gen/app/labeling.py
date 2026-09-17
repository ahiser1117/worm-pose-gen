"""Read-only labeling computations and explicit corpus traversal, independent of workspaces."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import binascii
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np

from ..corpus import CorpusStore, recording_identity
from ..label_app import Proposer, data_url, decode_mask_data_url, mask_to_png_values, probability_to_png, UNCERTAIN_BAND
from ..pipeline import workspace_dataset
from .state import NotFound, _integer


def encoded(mask):
    return data_url(mask_to_png_values(mask))


def decoded(value, shape):
    try:
        return decode_mask_data_url(str(value or ''), shape)
    except (OSError, binascii.Error) as error:
        raise ValueError('mask must contain a readable base64 PNG image') from error


def key(target):
    return f"{recording_identity(target['recording'], target['dataset'])}:{target['frame']}"


class LabelingService:
    def __init__(self, app):
        self.app = app
        self.manifests: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._fit_lock = threading.Lock()

    @property
    def store(self):
        return CorpusStore(self.app.config.corpus_root)

    def target(self, value):
        if not isinstance(value, dict):
            raise ValueError("target must identify a workspace, corpus sample, or recording frame")
        if value.get('workspace'):
            view = self.app.view(str(value['workspace']))
            frame = _integer(value, 'frame')
            view.workspace.row_of(frame)
            return {'workspace': str(value['workspace']), 'recording': str(view.workspace.recording.resolve()),
                    'dataset': workspace_dataset(view.workspace), 'frame': frame}
        if value.get('sample_id'):
            sample = self.store.get(str(value['sample_id']))
            if sample is None:
                raise NotFound('saved corpus label no longer exists')
            return {'sample_id': sample.sample_id, 'recording': str(Path(sample.source_path).resolve()),
                    'dataset': sample.dataset_path, 'frame': sample.frame_index}
        path = self.app.recording_path(str(value.get('recording') or ''))
        dataset = '/' + str(value.get('dataset') or self.app.recording_dataset(path)).strip('/')
        target = {'recording': str(path.resolve()), 'dataset': dataset, 'frame': _integer(value, 'frame')}
        source = self.source(target)
        if not 0 <= target['frame'] < source.frame_count:
            raise ValueError('frame is outside recording bounds')
        return target

    def source(self, target):
        source, error = self.app.viewer._source(target['recording'], target['dataset'])
        if source is None and Path(target['recording']).is_file():
            # A user may restore/register an initially missing source without restarting.
            cache_key = target['recording'] if target['dataset'] == '/img_nir' else f"{target['recording']}#{target['dataset']}"
            with self.app.viewer._lock:
                if self.app.viewer._sources.get(cache_key, (None,))[0] is None:
                    self.app.viewer._sources.pop(cache_key, None)
            source, error = self.app.viewer._source(target['recording'], target['dataset'])
        if source is None:
            raise ValueError(f"recording unavailable: {error}; locate and register the source HDF5 recording")
        return source

    def images(self, target):
        if target.get('sample_id'):
            with self.store.locked():
                image, mask, record = self.store.load(target['sample_id'])
                return self.store.load_raw(record.sample_id), image
        return self.source(target).corrected(target['frame'])

    def checkpoint(self, target):
        if target.get('workspace'):
            view = self.app.view(target['workspace'])
            return view.workspace.info.settings.get('checkpoint') or view.run.summary.get('selected_checkpoint') or (view.run.summary.get('checkpoint') or {}).get('path')
        return None if self.app.config.checkpoint is None else str(self.app.config.checkpoint)

    def matching(self, target):
        return self.store.find_frame(target['recording'], target['dataset'], target['frame'])

    def frame(self, value):
        target = self.target(value)
        raw, image = self.images(target)
        sample = self.matching(target)
        base = None
        revision = sample.revision if sample else 0
        mask = encoded(np.zeros(image.shape, dtype=np.uint8))
        override = False
        stale = False
        if target.get('workspace'):
            payload = self.app.view(target['workspace']).mask_payload(target['frame'], self.app.viewer.segmenters, self.app.device)
            mask, base, revision = payload['mask'], payload['base_mask'], payload['revision']
            override, stale = payload['has_override'], payload['stale']
        elif sample:
            _, label, _ = self.store.load(sample.sample_id)
            mask = encoded(label)
        pledge = self.store.frame_pledge(target['recording'], target['dataset'], target['frame'])
        manifest_id = value.get('manifest_id')
        queue_entry = None
        if manifest_id:
            queue_entry = next((e for e in self.manifest(manifest_id)['entries'] if key(e['target']) == key(target)), None)
            if queue_entry:
                target['manifest_id'] = manifest_id
                pledge = pledge or (None if queue_entry['split'] == 'auto' else queue_entry['split'])
        return {'target': target, 'key': key(target), 'frame': target['frame'], 'width': image.shape[1], 'height': image.shape[0],
                'image': data_url(image), 'image_raw': data_url(raw), 'mask': mask, 'base_mask': base,
                'has_override': override, 'stale': stale, 'revision': revision,
                'workspace_revision': revision if target.get('workspace') else None,
                'corpus_revision': sample.revision if sample else 0, 'sample': asdict(sample) if sample else None,
                'checkpoint': self.app.viewer.segmenters.signature(self.checkpoint(target)),
                'pledged_split': pledge, 'queue_entry': queue_entry, 'capabilities': {'network': self.app.viewer.segmenters.resolve(self.checkpoint(target)) is not None,
                    'classical': True, 'raw_threshold': True, 'saved_workspace': override, 'saved_corpus': sample is not None},
                'encoding': {'background': 0, 'worm': 255, 'ignore': 128}}

    def proposals(self, payload):
        target = self.target(payload.get('target'))
        source = payload.get('source')
        result = {'target': target, 'key': key(target), 'source': source, 'request_id': payload.get('request_id'),
                  'draft_generation': payload.get('draft_generation'), 'draft_revision': payload.get('draft_revision')}
        if source == 'saved_workspace':
            if not target.get('workspace'):
                raise ValueError('no workspace override is available for this target')
            workspace = self.app.workspace(target['workspace'])
            mask = workspace.get_override_mask(workspace.row_of(target['frame']))
            if mask is None:
                raise ValueError('no saved workspace override for this frame')
        elif source == 'saved_corpus':
            sample = self.matching(target)
            if sample is None:
                raise NotFound('no saved corpus label for this frame')
            _, mask, _ = self.store.load(sample.sample_id)
        elif source in ('network', 'classical', 'raw_threshold'):
            _, image = self.images(target)
            if source == 'network':
                probability, checkpoint = self.app.viewer.segmenters.probability(self.checkpoint(target), image)
                if probability is None:
                    raise ValueError('Network proposal requires an available checkpoint')
                threshold = float(payload.get('threshold', 0.5))
                if not 0 <= threshold <= 1:
                    raise ValueError('network threshold must be between 0 and 1')
                mask = (probability >= threshold).astype(np.uint8)
                result.update(probability=data_url(probability_to_png(probability)), checkpoint=checkpoint)
            else:
                classical, threshold = Proposer.classical(image)
                mask = (classical if source == 'classical' else threshold).astype(np.uint8)
        else:
            raise ValueError('unknown mask proposal source')
        return {**result, 'mask': encoded(mask)}

    def refine(self, payload):
        target = self.target(payload.get('target'))
        _, image = self.images(target)
        mask = decoded(payload.get('mask'), image.shape)
        method = str(payload.get('method') or '')
        # The tube fitter shares a device and is intentionally independent of pose jobs.
        with self._fit_lock:
            refined, info = Proposer.refine(mask, method, self.app.device)
        return {'target': target, 'key': key(target), 'mask': encoded(refined), 'info': info,
                'request_id': payload.get('request_id'), 'draft_generation': payload.get('draft_generation'),
                'draft_revision': payload.get('draft_revision')}

    def save(self, payload):
        target = self.target(payload.get('target'))
        if 'revision' not in payload:
            raise ValueError('corpus revision is required (0 for a new label)')
        revision = _integer(payload, 'revision')
        if revision < 0:
            raise ValueError('corpus revision must be nonnegative')
        raw, image = self.images(target)
        mask = decoded(payload.get('mask'), image.shape)
        split = payload.get('split')
        manifest_id = payload.get('manifest_id') or payload.get('target', {}).get('manifest_id')
        if manifest_id:
            manifest = self.manifest(manifest_id)
            entry = next((e for e in manifest['entries'] if key(e['target']) == key(target)), None)
            if entry is None:
                raise ValueError('target is not in the selected manifest')
            split = entry['split'] if entry['split'] != 'auto' else split
        if target.get('sample_id'):
            sample = self.store.update_label(target['sample_id'], mask, revision)
        else:
            sample = self.store.save_frame(target['recording'], target['dataset'], target['frame'], image, mask,
                image_raw=raw, revision=revision, split=split,
                label_source='manual:workspace' if target.get('workspace') else 'manual:corpus')
        return {'target': target, 'key': key(target), 'sample': asdict(sample), 'revision': sample.revision,
                'counts': self.store.counts(), 'root': str(self.store.root)}

    def load_manifest(self, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise NotFound('manifest file does not exist')
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('recordings'), dict) or not isinstance(data.get('frames'), list):
            raise ValueError('manifest needs a recordings object and a frames list')
        recordings, entries, errors = [], [], []
        aliases, pledges = {}, {}
        for alias, record in data['recordings'].items():
            if not isinstance(record, dict) or not isinstance(record.get('path'), str):
                raise ValueError('each manifest recording needs a path')
            source_path = Path(record['path']).expanduser()
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            source_path = source_path.resolve()
            dataset = '/' + str(record.get('dataset') or record.get('dataset_path') or '/img_nir').strip('/')
            split = record.get('split', 'auto')
            if split not in ('auto', 'train', 'val', 'test'):
                raise ValueError(f'unknown manifest split {split!r}')
            identity = recording_identity(source_path, dataset)
            if identity in pledges and pledges[identity] != split:
                raise ValueError('manifest aliases pledge the same recording to different splits')
            pledges[identity] = split
            record = {'recording': str(source_path), 'dataset': dataset, 'split': split, 'alias': alias}
            try:
                source = self.source(record)
                record['frame_count'] = source.frame_count
                # Registry access is sufficient; do not create or modify a workspace.
                self.app.registry.add(source_path, dataset)
            except (OSError, ValueError) as error:
                record['error'] = str(error)
                errors.append({'recording': alias, 'error': str(error)})
            aliases[alias] = record
            recordings.append(record)
        seen = set()
        for index, entry in enumerate(data['frames']):
            if not isinstance(entry, dict) or 'recording' not in entry or 'frame_index' not in entry:
                raise ValueError('each manifest frame needs recording and frame_index')
            if entry['recording'] not in aliases:
                raise ValueError('manifest frame refers to an unknown recording')
            record = aliases[entry['recording']]
            target = {k: record[k] for k in ('recording', 'dataset')}
            target['frame'] = int(entry['frame_index'])
            if target['frame'] < 0 or ('frame_count' in record and target['frame'] >= record['frame_count']):
                raise ValueError('manifest frame is outside recording bounds')
            if key(target) in seen:
                raise ValueError('manifest contains duplicate canonical frames')
            seen.add(key(target))
            entries.append({'target': target, 'split': record['split'], 'position': index + 1,
                            'reasons': entry.get('reasons', []), 'error': record.get('error')})
        manifest_id = hashlib.sha256((str(path) + json.dumps(data, sort_keys=True)).encode()).hexdigest()[:24]
        manifest = {'id': manifest_id, 'path': str(path), 'name': str(data.get('name', path.stem)),
                    'recordings': recordings, 'entries': entries, 'errors': errors}
        with self._lock:
            self.manifests[manifest_id] = manifest
        return self.manifest(manifest_id)

    def manifest(self, manifest_id):
        with self._lock:
            manifest = self.manifests.get(str(manifest_id))
        if manifest is None:
            raise NotFound('unknown manifest; load it again')
        labeled_keys = {f"{recording_identity(r.source_path, r.dataset_path)}:{r.frame_index}" for r in self.store.records()}
        labeled = sum(key(e['target']) in labeled_keys for e in manifest['entries'])
        return {**manifest, 'progress': {'total': len(manifest['entries']), 'labeled': labeled,
                                       'remaining': len(manifest['entries']) - labeled}}

    def list_manifests(self):
        with self._lock:
            ids = list(self.manifests)
        return [self.manifest(i) for i in ids]

    def pool(self, payload):
        pool = payload.get('pool') or {}
        if pool.get('workspace'):
            view = self.app.view(str(pool['workspace']))
            frames = pool.get('frames', view.workspace.frame_index.tolist())
            valid = set(map(int, view.workspace.frame_index))
            if not isinstance(frames, (list, tuple)) or any(int(f) not in valid for f in frames):
                raise ValueError('pool frames must be an explicit subset of the workspace')
            return [self.target({'workspace': pool['workspace'], 'frame': int(f)}) for f in sorted(set(map(int, frames)))]
        records = pool.get('recordings')
        if not isinstance(records, list) or not records:
            raise ValueError('declare a workspace or recording pool before navigating')
        result = []
        for record in records:
            first = int(record.get('first', 0))
            target = self.target({**record, 'frame': first})
            last = int(record.get('last', self.source(target).frame_count - 1))
            step = int(record.get('step', 1))
            if step <= 0 or last < first or last >= self.source(target).frame_count:
                raise ValueError('invalid recording pool bounds or step')
            result.extend({**target, 'frame': f} for f in range(first, last + 1, step))
        return list({key(t): t for t in result}.values())

    def next(self, payload):
        mode = payload.get('mode')
        current = payload.get('current')
        # A stable browse cursor remains valid even after its sample was deleted.
        current = self.target(current) if current and not (mode == 'browse' and payload.get('cursor')) else None
        current_key = key(current) if current else None
        labeled = {f"{recording_identity(r.source_path, r.dataset_path)}:{r.frame_index}" for r in self.store.records()}
        result: dict[str, Any] = {}
        if mode == 'browse':
            filters = payload.get('filters') or {}
            allowed = {k: filters.get(k, '') for k in ('source', 'split', 'recording', 'q')}
            records = self.store.filtered(**allowed)
            pool = payload.get('pool') or {}
            workspace_pool = None
            if pool.get('workspace'):
                workspace_pool = {key(t): t for t in self.pool(payload)}
                records = [r for r in records if f"{recording_identity(r.source_path, r.dataset_path)}:{r.frame_index}" in workspace_pool]
            elif pool.get('recordings'):
                # Filter canonical identities without opening missing saved sources.
                identities = {recording_identity(r['recording'], r.get('dataset', '/img_nir')) for r in pool['recordings']}
                records = [r for r in records if recording_identity(r.source_path, r.dataset_path) in identities]
            after = (payload.get('cursor') or {}).get('after')
            if not after and current:
                sample = self.matching(current)
                if sample:
                    after = [sample.source_path, sample.dataset_path, sample.frame_index, sample.sample_id]
            record = next((r for r in records if not after or (r.source_path, r.dataset_path, r.frame_index, r.sample_id) > tuple(after)), None)
            if record:
                target = self.target({'sample_id': record.sample_id})
                if workspace_pool is not None:
                    target = workspace_pool[key(target)]
                result = {'target': target,
                          'cursor': {'after': [record.source_path, record.dataset_path, record.frame_index, record.sample_id]}}
        elif mode == 'queue':
            manifest = self.manifest(payload.get('manifest_id'))
            entries = manifest['entries']
            if (payload.get('pool') or {}).get('workspace'):
                pool = {key(t): t for t in self.pool(payload)}
                entries = [{**e, 'target': pool[key(e['target'])]} for e in entries if key(e['target']) in pool]
            elif payload.get('pool'):
                pool_keys = {key(t) for t in self.pool(payload)}
                entries = [e for e in entries if key(e['target']) in pool_keys]
            start = next((i + 1 for i, e in enumerate(entries) if key(e['target']) == current_key), 0)
            pending = [e for e in entries[start:] if key(e['target']) not in labeled]
            if pending:
                entry = pending[0]
                if entry.get('error'):
                    return {'target': None, 'exhausted': False, 'blocked': True, 'reason': entry['error'], 'progress': manifest['progress']}
                result = {'target': {**entry['target'], 'manifest_id': manifest['id']}, 'queue_entry': entry, 'pledged_split': entry['split'], 'progress': manifest['progress']}
            else:
                result['progress'] = manifest['progress']
        elif mode in ('sequential', 'random', 'uncertain'):
            pool = self.pool(payload)
            if mode == 'sequential':
                stride = _integer(payload, 'stride', 1)
                if stride < 1:
                    raise ValueError('stride must be positive')
                index = next((i for i, t in enumerate(pool) if key(t) == current_key), None)
                position = 0 if index is None else index + stride
                if position < len(pool):
                    result['target'] = pool[position]
            else:
                if mode == 'uncertain' and self.app.viewer.segmenters.resolve(self.checkpoint(pool[0] if pool else {})) is None:
                    raise ValueError('Network-uncertain mode requires an available checkpoint')
                pending = [t for t in pool if key(t) not in labeled and key(t) != current_key]
                if pending:
                    rng = np.random.default_rng()
                    if mode == 'random':
                        result['target'] = pending[int(rng.integers(len(pending)))]
                    else:
                        count = _integer(payload, 'candidates', 16)
                        if not 1 <= count <= 256:
                            raise ValueError('uncertainty candidate count must be between 1 and 256')
                        best = None
                        for i in rng.choice(len(pending), min(count, len(pending)), replace=False):
                            target = pending[int(i)]
                            _, image = self.images(target)
                            probability, _ = self.app.viewer.segmenters.probability(self.checkpoint(target), image)
                            if probability is None:
                                raise ValueError('Network-uncertain mode requires an available checkpoint')
                            score = float(((probability > UNCERTAIN_BAND[0]) & (probability < UNCERTAIN_BAND[1])).mean())
                            if best is None or score > best[0]:
                                best = score, target
                        result.update(target=best[1], uncertainty=best[0])
        else:
            raise ValueError('unknown next-frame mode')
        if result.get('target'):
            return {**result, 'key': key(result['target']), 'exhausted': False}
        return {**result, 'target': None, 'exhausted': True, 'reason': 'No more matching frames in the declared pool'}
