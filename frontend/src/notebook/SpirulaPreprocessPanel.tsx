import { useEffect, useState } from 'react';
import { generatePreprocessNotebook, getPreprocessCatalog, validatePreprocessPreset } from '../api';
import { resolveDriveFolder } from './drivePath';
import { NotebookDownload } from './NotebookDownload';
import type { GeneratedNotebook } from './types';
import type { PreprocessCatalog, PreprocessField, PreprocessInput, PreprocessSpec, SettingValue } from './preprocessTypes';
import './preprocess.css';

const LABELS: Record<string, string> = {
  low: 'Düşük', medium: 'Orta', high: 'Yüksek (önerilen)', extreme: 'En yüksek',
  auto: 'Otomatik', single: 'Tek kamera', folder: 'Klasör başına bir kamera', image: 'Görüntü başına bir kamera',
  exhaustive: 'Tüm çiftler', sequential: 'Sıralı', prefilter: 'GPU ile çift seçimi',
  video: 'Video kareleri', individual: 'Tek tek fotoğraflar', internet: 'Sırasız koleksiyon',
  mapping: 'Haritalama sırasında', final: 'Yalnız son geçişte', fixed: 'Sabit tut',
  'moge2-vits': 'MoGe-2 küçük', 'moge2-vitb': 'MoGe-2 orta', 'moge2-vitl': 'MoGe-2 büyük',
  'metric3d-vit-small': 'Metric3D küçük', 'metric3d-vit-large': 'Metric3D büyük', 'metric3d-vit-giant2': 'Metric3D dev',
  'sam3-q4_0': 'SAM 3 (Q4 · 707 MB)', 'sam3-f16': 'SAM 3 (F16 · 1.84 GB)',
  none: 'Kapalı', up: 'Yalnız yukarı yönü', horizontal: 'Enlem ve boylam', full: 'Enlem, boylam ve yükseklik',
};

function Setting({ name, field, value, onChange }: {
  name: string; field: PreprocessField; value: SettingValue; onChange(value: SettingValue): void;
}) {
  const id = `prep-${name}`;
  const help = field.description ? `${id}-help` : undefined;
  if (field.type === 'boolean') return <div className="prep-setting prep-check">
    <label htmlFor={id}><input id={id} type="checkbox" checked={Boolean(value)} onChange={e => onChange(e.target.checked)} aria-describedby={help} />{field.title}</label>
    {help && <small id={help}>{field.description}</small>}
  </div>;
  return <div className="prep-setting">
    <label htmlFor={id}>{field.title}</label>
    {field.enum ? <select id={id} value={String(value)} onChange={e => onChange(e.target.value)} aria-describedby={help}>
      {field.enum.map(option => <option key={option} value={option}>{LABELS[option] ?? option}</option>)}
    </select> : <input id={id} value={String(value)} type={['number', 'integer'].includes(field.type) ? 'number' : 'text'}
      min={field.minimum} max={field.maximum} maxLength={field.maxLength} step={field.type === 'integer' ? 1 : 'any'}
      required={['number', 'integer'].includes(field.type)} aria-describedby={help}
      onChange={e => onChange(e.target.type === 'number' && e.target.value !== '' ? Number(e.target.value) : e.target.value)} />}
    {help && <small id={help}>{field.description}</small>}
  </div>;
}

export function SpirulaPreprocessPanel() {
  const [catalog, setCatalog] = useState<PreprocessCatalog | null>(null);
  const [settings, setSettings] = useState<Record<string, SettingValue>>({});
  const [inputs, setInputs] = useState<PreprocessInput[]>([{ kind: 'video', path: '', fps: 4 }]);
  const [outputPath, setOutputPath] = useState('GaussianTests/datasets');
  const [datasetName, setDatasetName] = useState('spirula_dataset');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState(0);
  const [artifact, setArtifact] = useState<GeneratedNotebook | null>(null);
  const [notice, setNotice] = useState('');
  useEffect(() => {
    let active = true;
    setError('');
    getPreprocessCatalog().then(value => {
      if (active) { setCatalog(value); setSettings(value.defaults); }
    }, err => { if (active) setError(String(err)); });
    return () => { active = false; };
  }, [retry]);

  const paths = inputs.map(input => resolveDriveFolder(input.path));
  const output = resolveDriveFolder(outputPath);
  const validPaths = paths.every(path => path.ok) && output.ok && /^[A-Za-z0-9][A-Za-z0-9_-]{0,59}$/.test(datasetName);
  function makeSpec(): PreprocessSpec {
    return { schema_version: 1, inputs: inputs.map((input, i) => ({ ...input, path: paths[i].ok ? paths[i].canonicalInput : input.path })),
      dataset_name: datasetName, output_path: output.ok ? output.canonicalInput : outputPath, settings };
  }
  function editInput(i: number, patch: Partial<PreprocessInput>) {
    setInputs(current => current.map((entry, n) => n === i ? { ...entry, ...patch } : entry));
    setArtifact(null);
  }
  async function generate() {
    if (!validPaths) return;
    setBusy(true); setError('');
    try { setArtifact(await generatePreprocessNotebook(makeSpec())); }
    catch (err) { setError(String(err)); }
    finally { setBusy(false); }
  }
  async function savePreset() {
    setBusy(true); setError('');
    try {
      const validated = await validatePreprocessPreset(makeSpec());
      const url = URL.createObjectURL(new Blob([JSON.stringify(validated, null, 2)], { type: 'application/json' }));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${datasetName}_preprocess.json`; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setNotice('Hazır ayar indirildi. Bu uygulamada yeniden yükleyebilirsin.');
    } catch (err) { setError(String(err)); }
    finally { setBusy(false); }
  }
  async function loadPreset(file: File) {
    setBusy(true); setError('');
    try {
      if (file.size > 1024 * 1024) throw new Error('Hazır ayar dosyası en fazla 1 MB olabilir.');
      const validated = await validatePreprocessPreset(JSON.parse(await file.text()));
      setSettings(validated.settings); setInputs(validated.inputs); setOutputPath(validated.output_path);
      setDatasetName(validated.dataset_name); setArtifact(null); setNotice('Hazır ayar yüklendi. Giriş yollarını kontrol et.');
    } catch (err) { setError(String(err)); }
    finally { setBusy(false); }
  }
  if (!catalog) return <div className="prep-panel"><p role={error ? 'alert' : 'status'}>{error || 'Spirula seçenekleri yükleniyor…'}</p>
    {error && <button onClick={() => setRetry(retry + 1)}>Yeniden dene</button>}</div>;

  return <main className="prep-panel">
    <header className="prep-header"><h1>Spirula dataset hazırlama</h1><p>Ayarlarını seç, notebook’u indir ve Colab’de çalıştır.</p></header>
    <form onSubmit={e => { e.preventDefault(); void generate(); }} onChange={() => { setArtifact(null); setNotice(''); }}>
      <fieldset disabled={busy} className="prep-controls">
        <legend className="prep-sr-only">Dataset ayarları</legend>
        <section className="prep-section"><h2>Kaynaklar</h2>
          {inputs.map((input, i) => <div className="prep-source" key={i}>
            <label className="prep-source-path">{input.kind === 'video' ? 'Video' : 'Fotoğraf klasörü'} {i + 1}
              <input aria-label={`Kaynak ${i + 1} MyDrive yolu`} value={input.path} required placeholder="GaussianTests/inputs/IMG_5966.MOV"
                onChange={e => editInput(i, { path: e.target.value })} aria-invalid={Boolean(input.path && !paths[i].ok)} /></label>
            {input.kind === 'video' && <label>FPS<input type="number" min="0.01" max="120" step="any" required aria-label={`Kaynak ${i + 1} FPS`}
              value={input.fps || ''} onChange={e => editInput(i, { fps: Number(e.target.value) })} /></label>}
            <button type="button" disabled={inputs.length === 1} onClick={() => { setInputs(inputs.filter((_, n) => n !== i)); setArtifact(null); }}>Kaldır {i + 1}</button>
            {input.path && !paths[i].ok && <small className="prep-error">{paths[i].error}</small>}
          </div>)}
          <div className="prep-actions">
            <button type="button" disabled={inputs.length >= 32} onClick={() => { setInputs([...inputs, { kind: 'video', path: '', fps: 4 }]); setArtifact(null); }}>Video ekle</button>
            <button type="button" disabled={inputs.length >= 32} onClick={() => { setInputs([...inputs, { kind: 'photos', path: '', fps: 4 }]); setArtifact(null); }}>Fotoğraf klasörü ekle</button>
          </div>
          <p className="prep-help">Dosyaları önce Drive’a yükle. Yollar MyDrive’a göredir. Fotoğraf klasöründe JPG, PNG veya BMP kullan.</p>
          <div className="prep-output-row">
            <label>Drive çıktı klasörü<input value={outputPath} required onChange={e => setOutputPath(e.target.value)} /></label>
            <label>Dataset adı<input value={datasetName} required pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,59}" onChange={e => setDatasetName(e.target.value)} /></label>
          </div>
          {!output.ok && <p className="prep-error">{output.error}</p>}
        </section>
        <section className="prep-section"><h2>Hazır ayar</h2>
          <div className="prep-actions">
            <button type="button" onClick={() => { setSettings(catalog.defaults); setArtifact(null); setNotice('Genel ayarlar uygulandı.'); }}>Genel ayarlara dön</button>
            <button type="button" disabled={!validPaths} onClick={() => void savePreset()}>Hazır ayar olarak kaydet</button>
            <label className="prep-file">Hazır ayar yükle<input aria-label="Hazır ayar yükle" type="file" accept=".json,application/json" onChange={e => {
              const file = e.target.files?.[0]; if (file) void loadPreset(file); e.target.value = '';
            }} /></label>
          </div>
          <p className="prep-help">Bu uygulamanın ayar dosyaları kullanılır. Spirula’nın masaüstü hazır ayar dosyaları ayrı formattadır.</p>
        </section>
        {catalog.groups.map(group => <details className="prep-section" key={group} open={group === 'Ayarlar' || group === 'Geometri'}>
          <summary>{group}</summary><div className="prep-grid">
            {Object.entries(catalog.fields).filter(([, field]) => field.section === group && (!field.when || settings[field.when])).map(([name, field]) =>
              <Setting key={name} name={name} field={field} value={settings[name] ?? field.default} onChange={value => setSettings({ ...settings, [name]: value })} />)}
          </div>
          {group === 'Maskeler' && <p className="prep-help">Metinle nesne seçimi ve sabit şekiller desteklenir. Fotoğraflar ayrı maskelenir; bu sürümde model her fotoğrafta yeniden yüklenir. Görüntü üzerinde fırça veya tıklama ile maske çizimi bu ekranda yok.</p>}
        </details>)}
        <footer className="prep-footer">
          <div><strong>Dataset ZIP + kalite raporu</strong><p>images, sparse ve seçtiğin haritalar kaydedilir. Eğitim daha sonra seçilir.</p>
            <small>Aygıt: Colab NVIDIA GPU · Her çalışma yeni klasöre kaydedilir.</small></div>
          <button type="submit" className="prep-primary" disabled={!validPaths || busy}>{busy ? 'Hazırlanıyor…' : 'Preprocessing notebook’unu indir'}</button>
        </footer>
      </fieldset>
    </form>
    {notice && <p role="status">{notice}</p>}
    {error && <p role="alert" className="prep-error">{error}</p>}
    {artifact && <NotebookDownload artifact={artifact} onRegenerate={() => void generate()} onEdit={() => setArtifact(null)} />}
    <p className="prep-credit">Dataset hazırlama motoru: <a href="https://github.com/harry7557558/spirula-studio" target="_blank" rel="noreferrer">Spirula Studio {catalog.spirula_version}</a></p>
  </main>;
}
