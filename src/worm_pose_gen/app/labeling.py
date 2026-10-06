"""Mask labeling for the corpus and for workspace frames: frames, proposals, refinement, saves, and label groups.

A *target* is a workspace frame (``workspace`` + ``frame``: the workspace's
mask correction, which feeds fitting), a saved corpus label (``sample_id``),
or a recording frame (``recording`` + ``dataset`` + ``frame``).  Paint, the
corpus labeling section of the app, walks a *label group*: an ordered list of
targets with optional split pledges.  A group is one of

``manifest``  a labeling manifest file (``recordings`` aliases with split
              pledges, ``frames``); the repository's ``docs/labeling_*/manifest.json``
              are offered for loading;
``section``   a stretch of one recording chosen for relabeling (``first``..``last``
              by ``step``), usually sent from a workspace's Inspect selection;
              sections persist in ``<workspaces_root>/label_sections.json``;
``samples``   saved corpus labels (an opened label, the Labels filter, Body fields).

Group progress counts the entries that have a corpus label.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import binascii
import json
from pathlib import Path
import threading

import numpy as np

from ..corpus import CorpusStore, recording_identity
from ..label_app import Proposer, data_url, decode_mask_data_url, mask_to_png_values, probability_to_png
from ..jobs import REPO_ROOT
from ..pipeline import workspace_dataset
from ..workspace import _write_json_atomic, utc_now
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


MANIFEST_GLOB = "docs/labeling_*/manifest.json"
GROUP_KINDS = ('manifest', 'section', 'samples')


class LabelingService:
    def __init__(self, app):
        self.app = app
        self.groups: dict[str, dict] = {}
        self.sections_path = app.config.workspaces_root / 'label_sections.json'
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

    def frame(self, value, group_id=None):
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
        entry = None
        if group_id:
            entry = self.entry(group_id, target)
            pledge = pledge or (None if entry['split'] == 'auto' else entry['split'])
        return {'target': target, 'key': key(target), 'frame': target['frame'], 'width': image.shape[1], 'height': image.shape[0],
                'image': data_url(image), 'image_raw': data_url(raw), 'mask': mask, 'base_mask': base,
                'has_override': override, 'stale': stale, 'revision': revision,
                'workspace_revision': revision if target.get('workspace') else None,
                'corpus_revision': sample.revision if sample else 0, 'sample': asdict(sample) if sample else None,
                'checkpoint': self.app.viewer.segmenters.signature(self.checkpoint(target)),
                'pledged_split': pledge, 'entry': entry, 'capabilities': {'network': self.app.viewer.segmenters.resolve(self.checkpoint(target)) is not None,
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
        if payload.get('group_id'):
            entry = self.entry(payload['group_id'], target)
            split = entry['split'] if entry['split'] != 'auto' else split
        if target.get('sample_id'):
            sample = self.store.update_label(target['sample_id'], mask, revision)
        else:
            sample = self.store.save_frame(target['recording'], target['dataset'], target['frame'], image, mask,
                image_raw=raw, revision=revision, split=split,
                label_source='manual:workspace' if target.get('workspace') else 'manual:corpus')
        return {'target': target, 'key': key(target), 'sample': asdict(sample), 'revision': sample.revision,
                'counts': self.store.counts(), 'root': str(self.store.root)}

    # ----------------------------------------------------------------- groups

    def entry(self, group_id, target):
        entry = next((e for e in self._group(group_id)['entries'] if key(e['target']) == key(target)), None)
        if entry is None:
            raise ValueError('target is not in the selected label group')
        return entry

    def _group(self, group_id):
        self._load_sections()
        with self._lock:
            group = self.groups.get(str(group_id))
        if group is None:
            raise NotFound('unknown label group; open it again')
        return group

    def _labeled(self):
        return {f"{recording_identity(r.source_path, r.dataset_path)}:{r.frame_index}" for r in self.store.records()}

    def _summary(self, group, labeled):
        done = sum(key(e['target']) in labeled for e in group['entries'])
        return {k: v for k, v in group.items() if k != 'entries'} | {
            'progress': {'total': len(group['entries']), 'labeled': done, 'remaining': len(group['entries']) - done}}

    def group(self, group_id):
        """A group with every entry marked ``labeled`` and the position of its first unlabeled entry (or None)."""

        group, labeled = self._group(group_id), self._labeled()
        entries = [{**e, 'labeled': key(e['target']) in labeled} for e in group['entries']]
        return {**self._summary(group, labeled), 'entries': entries,
                'first_unlabeled': next((i for i, e in enumerate(entries) if not e['labeled']), None)}

    def list_groups(self):
        """Opened groups with progress, and the repository's manifests not yet opened."""

        self._load_sections()
        labeled = self._labeled()
        with self._lock:
            groups = list(self.groups.values())
        opened = {g.get('path') for g in groups if g['kind'] == 'manifest'}
        discovered = []
        for path in sorted(REPO_ROOT.glob(MANIFEST_GLOB)):
            if str(path.resolve()) in opened:
                continue
            try:
                data = json.loads(path.read_text())
                discovered.append({'path': str(path.resolve()), 'name': str(data.get('name', path.parent.name)),
                                   'description': str(data.get('description', '')), 'frames': len(data.get('frames', []))})
            except (OSError, ValueError) as error:
                discovered.append({'path': str(path.resolve()), 'name': path.parent.name, 'error': str(error)})
        return {'groups': [self._summary(g, labeled) for g in groups], 'discovered': discovered}

    def open_group(self, payload):
        kind = payload.get('kind')
        if kind == 'manifest':
            return self.load_manifest(str(payload.get('path') or ''))
        if kind == 'section':
            return self.create_section(payload)
        if kind == 'samples':
            return self.sample_group(payload.get('sample_ids'), str(payload.get('name') or ''))
        raise ValueError(f'unknown label group kind {kind!r}; expected one of {GROUP_KINDS}')

    def _register(self, group):
        with self._lock:
            self.groups[group['id']] = group
        return self.group(group['id'])

    def close_group(self, group_id):
        """Forget a group; a section is also removed from the saved sections."""

        group = self._group(group_id)
        if group['kind'] == 'section':
            with self._lock:
                sections = self._read_sections()
                sections.pop(group['id'], None)
                _write_json_atomic(self.sections_path, sections)
        with self._lock:
            self.groups.pop(group['id'], None)
        return {'closed': group['id']}

    def sample_group(self, sample_ids, name=''):
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError('a samples group needs a non-empty sample_ids list')
        entries = []
        for index, sample_id in enumerate(dict.fromkeys(map(str, sample_ids))):
            target = self.target({'sample_id': sample_id})
            entries.append({'target': target, 'split': 'auto', 'position': index + 1, 'reasons': [], 'error': None})
        group_id = 'smp_' + hashlib.sha256(json.dumps([e['target']['sample_id'] for e in entries]).encode()).hexdigest()[:20]
        label = name or (entries[0]['target']['sample_id'] if len(entries) == 1 else f'{len(entries)} saved labels')
        return self._register({'id': group_id, 'kind': 'samples', 'name': label, 'entries': entries, 'errors': []})

    def _read_sections(self):
        return json.loads(self.sections_path.read_text()) if self.sections_path.exists() else {}

    def _section_group(self, section_id, section):
        target = {'recording': section['recording'], 'dataset': section['dataset']}
        entries = [{'target': {**target, 'frame': frame}, 'split': 'auto', 'position': index + 1, 'reasons': [], 'error': None}
                   for index, frame in enumerate(range(section['first'], section['last'] + 1, section['step']))]
        return {'id': section_id, 'kind': 'section', 'entries': entries, 'errors': [], **section}

    def _load_sections(self):
        with self._lock:
            for section_id, section in self._read_sections().items():
                if section_id not in self.groups:
                    self.groups[section_id] = self._section_group(section_id, section)

    def create_section(self, payload):
        """Persist a recording stretch to relabel; the same recording, range and step give the same section."""

        first, last = _integer(payload, 'first'), _integer(payload, 'last')
        if payload.get('workspace'):
            # A range selected in a workspace: its recording, dataset and frame stride; the workspace is not kept.
            workspace = self.app.workspace(str(payload['workspace']))
            frames = workspace.frame_index
            stride = int(frames[1] - frames[0]) if len(frames) > 1 else 1
            payload = {**payload, 'recording': str(workspace.recording), 'dataset': workspace_dataset(workspace),
                       'step': payload.get('step') or stride, 'origin': payload.get('origin') or {'workspace': str(payload['workspace'])},
                       'name': payload.get('name') or f"{payload['workspace']} frames {first}-{last}"}
        step = _integer(payload, 'step', 1)
        target = self.target({'recording': payload.get('recording'), 'dataset': payload.get('dataset'), 'frame': first})
        frames = self.source(target).frame_count
        if step < 1 or last < first or last >= frames:
            raise ValueError(f'a section needs 0 <= first <= last < {frames} and a positive step')
        identity = json.dumps([target['recording'], target['dataset'], first, last, step])
        section_id = 'sec_' + hashlib.sha256(identity.encode()).hexdigest()[:20]
        name = str(payload.get('name') or f"{Path(target['recording']).stem} frames {first}-{last}")
        section = {'name': name, 'recording': target['recording'], 'dataset': target['dataset'], 'first': first,
                   'last': last, 'step': step, 'origin': payload.get('origin'), 'created_at': utc_now()}
        with self._lock:
            sections = self._read_sections()
            section = sections.setdefault(section_id, section)
            _write_json_atomic(self.sections_path, sections)
            self.groups[section_id] = self._section_group(section_id, section)
        return self.group(section_id)

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
        return self._register({'id': manifest_id, 'kind': 'manifest', 'path': str(path), 'name': str(data.get('name', path.stem)),
                               'description': str(data.get('description', '')), 'recordings': recordings,
                               'entries': entries, 'errors': errors})

