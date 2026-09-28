import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { SpirulaPreprocessPanel } from '../SpirulaPreprocessPanel';
import { generatePreprocessNotebook, getPreprocessCatalog, validatePreprocessPreset } from '../../api';

vi.mock('../../api', () => ({ generatePreprocessNotebook: vi.fn(), getPreprocessCatalog: vi.fn(), validatePreprocessPreset: vi.fn() }));

const catalog = {
  schema_version: 1, spirula_version: '2026.9.24', groups: ['Ayarlar', 'Geometri'],
  defaults: { quality: 'high', generate_geometry: true, geometry_model: 'moge2-vitb' },
  fields: {
    quality: { title: 'Kalite', section: 'Ayarlar', type: 'string', enum: ['high', 'extreme'], default: 'high' },
    generate_geometry: { title: 'Derinlik ve normalleri kestir', section: 'Geometri', type: 'boolean', default: true },
    geometry_model: { title: 'Geometri modeli', section: 'Geometri', type: 'string', enum: ['moge2-vitb', 'moge2-vitl'], default: 'moge2-vitb', when: 'generate_geometry' },
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getPreprocessCatalog).mockResolvedValue(catalog);
  vi.mocked(generatePreprocessNotebook).mockResolvedValue({ blob: new Blob(['notebook']), filename: 'room_preprocessing.ipynb' });
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test') });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
});

it('downloads selected preprocessing settings and multiple inputs without training fields', async () => {
  render(<SpirulaPreprocessPanel />);
  await screen.findByLabelText('Kalite');
  fireEvent.change(screen.getByLabelText('Kaynak 1 MyDrive yolu'), { target: { value: 'captures/room.MOV' } });
  fireEvent.change(screen.getByLabelText('Kalite'), { target: { value: 'extreme' } });
  fireEvent.click(screen.getByRole('button', { name: 'Fotoğraf klasörü ekle' }));
  fireEvent.change(screen.getByLabelText('Kaynak 2 MyDrive yolu'), { target: { value: 'captures/photos' } });
  fireEvent.click(screen.getByLabelText('Derinlik ve normalleri kestir'));
  expect(screen.queryByLabelText('Geometri modeli')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Preprocessing notebook’unu indir' }));
  await waitFor(() => expect(generatePreprocessNotebook).toHaveBeenCalledOnce());
  const payload = vi.mocked(generatePreprocessNotebook).mock.calls[0][0];
  expect(payload.inputs).toEqual([{ kind: 'video', path: 'captures/room.MOV', fps: 4 }, { kind: 'photos', path: 'captures/photos', fps: 4 }]);
  expect(payload.settings).toEqual({ quality: 'extreme', generate_geometry: false, geometry_model: 'moge2-vitb' });
  expect(payload).not.toHaveProperty('iterations');
  expect(await screen.findByText('room_preprocessing.ipynb')).toBeVisible();
});

it('rejects a bad Drive path before generating and offers catalog retry', async () => {
  vi.mocked(getPreprocessCatalog).mockRejectedValueOnce(new Error('offline'));
  render(<SpirulaPreprocessPanel />);
  await screen.findByText(/offline/);
  fireEvent.click(screen.getByRole('button', { name: 'Yeniden dene' }));
  await screen.findByLabelText('Kalite');
  fireEvent.change(screen.getByLabelText('Kaynak 1 MyDrive yolu'), { target: { value: '../bad' } });
  expect(screen.getByRole('button', { name: 'Preprocessing notebook’unu indir' })).toBeDisabled();
  expect(generatePreprocessNotebook).not.toHaveBeenCalled();
});

it('does not replace current settings with an invalid imported preset', async () => {
  vi.mocked(validatePreprocessPreset).mockRejectedValueOnce(new Error('Bilinmeyen ayar'));
  render(<SpirulaPreprocessPanel />);
  await screen.findByLabelText('Kalite');
  const file = { text: async () => JSON.stringify({ settings: { arbitrary: true } }) };
  fireEvent.change(screen.getByLabelText('Hazır ayar yükle'), { target: { files: [file] } });
  await screen.findByText(/Bilinmeyen ayar/);
  expect(screen.getByLabelText('Kalite')).toHaveValue('high');
});
