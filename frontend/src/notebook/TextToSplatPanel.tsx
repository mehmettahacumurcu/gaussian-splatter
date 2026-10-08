import { useEffect, useRef, useState } from "react";
import { generatePipelineNotebook, getPipelineNotebookPresets } from "../api";
import { resolveDriveFolder } from "./drivePath";
import { NotebookDownload } from "./NotebookDownload";
import type { PipelinePreset, PipelinePresets, TextToSplatNotebookSpec } from "./PipelineNotebookPanel";
import { BACKGROUND_MODELS, GPU_PRESETS, IMAGE_MODELS, RECONSTRUCTION_MODELS } from "./textToSplatSettings";
import type { BackgroundModel, GpuPreset, ImageModel, ReconstructionModel, TextToSplatModelSettings } from "./textToSplatSettings";
import type { GeneratedNotebook } from "./types";

function integerInRange(value: string, minimum: number, maximum: number): boolean {
  return value.trim() !== "" && Number.isInteger(Number(value)) && Number(value) >= minimum && Number(value) <= maximum;
}

function inRange(value: number, min: number, max: number, integer = true): boolean {
  return Number.isFinite(value) && (!integer || Number.isInteger(value)) && value >= min && value <= max;
}

type NumericSetting = { [K in keyof TextToSplatModelSettings]: TextToSplatModelSettings[K] extends number ? K : never }[keyof TextToSplatModelSettings];

function NumberSetting({ name, label, help, value, min, max, fractional = false, disabled = false, onChange }: {
  name: NumericSetting; label: string; help: string; value: number; min: number; max: number;
  fractional?: boolean; disabled?: boolean; onChange: (name: NumericSetting, value: number) => void;
}) {
  return <div>
    <label>{label}<input type="number" min={min} max={max} step={fractional ? "any" : "1"}
      required disabled={disabled} value={Number.isFinite(value) ? value : ""}
      aria-invalid={!inRange(value, min, max, !fractional)} aria-describedby={`text-splat-${name}-help`}
      onChange={event => onChange(name, event.target.value.trim() === "" ? NaN : Number(event.target.value))} /></label>
    <p id={`text-splat-${name}-help`} className="nb-help">{help}</p>
  </div>;
}

export function TextToSplatPanel() {
  const [catalog, setCatalog] = useState<PipelinePresets | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [retry, setRetry] = useState(0);
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [style, setStyle] = useState("");
  const [seed, setSeed] = useState("42");
  const [preset, setPreset] = useState<PipelinePreset>("baseline");
  const [settings, setSettings] = useState<TextToSplatModelSettings>({ ...GPU_PRESETS.l4.settings });
  const [outputDir, setOutputDir] = useState("");
  const [targetCount, setTargetCount] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [artifact, setArtifact] = useState<GeneratedNotebook | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);
  useEffect(() => {
    let active = true;
    setCatalogError("");
    getPipelineNotebookPresets().then(value => {
      if (!active) return;
      if (!value.presets.text_to_splat) {
        setCatalogError("Metinden splat ayarları bulunamadı. Backend sürümünü kontrol et.");
      } else setCatalog(value);
    }, err => { if (active) setCatalogError(`Hazır ayarlar yüklenemedi: ${String(err)}`); });
    return () => { active = false; };
  }, [retry]);

  const defaults = catalog?.presets.text_to_splat?.[preset];
  const imageModel = IMAGE_MODELS[settings.image_model];
  const gpu = GPU_PRESETS[settings.gpu_preset];
  const meshRoute = settings.reconstruction_model === "trellis2";
  const minimumVram = Math.max(imageModel.vram, meshRoute ? 28 + (settings.mesh_splat_cap > 150000 ? 4 : 0) : 20,
    preset === "ultra" ? 38 : 0);
  const gpuError = minimumVram > gpu.capacity
    ? `Seçilen modeller için en az ${minimumVram} GiB gerekir; ${gpu.label} profili yeterli değil. Daha büyük GPU profili veya daha hafif model seç.`
    : "";
  const blackwellWarning = settings.gpu_preset === "rtx_pro_6000" && !meshRoute;
  const output = outputDir.trim() ? resolveDriveFolder(outputDir, { allowResultFolder: true }) : null;
  const promptValid = prompt.trim().length >= 1 && Array.from(prompt.trim()).length <= 2000 && !prompt.includes("\0");
  const seedValid = integerInRange(seed, 0, 2147483647);
  const countValid = meshRoute || !targetCount.trim() || integerInRange(targetCount, 1000, 2000000);
  const imageStepMax = settings.image_model === "flux1_schnell" ? 4 : 100;
  const settingsValid = inRange(settings.image_steps, 1, imageStepMax) && inRange(settings.image_guidance, 0, 20, false)
    && [512, 768, 1024].includes(settings.image_resolution) && inRange(settings.trellis_seed, 0, 2147483647)
    && inRange(settings.sparse_steps, 1, 100) && inRange(settings.sparse_cfg, 0, 20, false)
    && inRange(settings.slat_steps, 1, 100) && inRange(settings.slat_cfg, 0, 20, false)
    && inRange(settings.mesh_views, 12, 120) && inRange(settings.mesh_fit_iterations, 100, 10000)
    && inRange(settings.mesh_splat_cap, 1000, 300000) && settings.reconstruction_model !== "hunyuan3d";
  const valid = promptValid && seedValid && countValid && settingsValid && !gpuError && (!output || output.ok)
    && (!imageModel.negative || (Array.from(negative).length <= 2000 && !negative.includes("\0")))
    && Array.from(style).length <= 500 && !style.includes("\0") && Boolean(defaults);

  function chooseGpu(value: string) {
    if (!(value in GPU_PRESETS)) return;
    setSettings({ ...GPU_PRESETS[value as GpuPreset].settings });
    setPreset("baseline");
  }
  function chooseImage(value: string) {
    if (!(value in IMAGE_MODELS)) return;
    const model = IMAGE_MODELS[value as ImageModel];
    setSettings(previous => ({ ...previous, image_model: value as ImageModel, image_steps: model.steps, image_guidance: model.guidance }));
  }
  function chooseQuality(value: PipelinePreset) {
    setPreset(value);
    const recipe = catalog?.presets.text_to_splat[value];
    if (recipe) setSettings(previous => ({ ...previous, sparse_steps: recipe.sparse_steps, slat_steps: recipe.slat_steps,
      image_steps: previous.image_model === "sdxl" ? recipe.image_steps : previous.image_steps }));
  }
  function setNumber(name: NumericSetting, value: number) {
    setSettings(previous => ({ ...previous, [name]: value }));
  }

  async function generate() {
    if (!valid || loading) return;
    const spec: TextToSplatNotebookSpec = {
      pipeline: "text_to_splat", input_mode: "text", input_path: "",
      prompt: prompt.trim(), negative_prompt: imageModel.negative ? negative.trim() : "", style: style.trim(),
      seed: Number(seed), preset, output_dir: output?.ok ? output.canonicalInput : null,
      target_splat_count: !meshRoute && targetCount.trim() ? Number(targetCount) : null,
      ...settings,
    };
    setLoading(true); setError(""); setArtifact(null);
    try {
      const result = await generatePipelineNotebook(spec);
      if (alive.current) setArtifact(result);
    } catch (err) {
      if (alive.current) setError(`Notebook oluşturulamadı: ${String(err)}`);
    } finally {
      if (alive.current) setLoading(false);
    }
  }

  return <div className="notebook-generator">
    <header className="nb-hero">
      <p className="nb-eyebrow">Morph için hedef B</p>
      <h1>Metinden hedef splat.</h1>
      <p>Bir nesne tarif et. Colab notebook’u seçtiğin modelle görsel üretir, arka planı ayırır ve 3DGS PLY’ye dönüştürür. Sonucu Scene Editor’a ekleyip morph hedefi olarak seçebilirsin.</p>
    </header>
    {catalogError && <div className="nb-generation-error" role="alert"><p>{catalogError}</p>
      <button type="button" onClick={() => setRetry(value => value + 1)}>Yeniden dene</button>
    </div>}
    {!catalog && !catalogError && <p role="status">Hazır ayarlar yükleniyor…</p>}
    <form className="nb-flow" onSubmit={event => { event.preventDefault(); void generate(); }} onChange={() => setArtifact(null)}>
      <fieldset disabled={loading} className="nb-stage-body nb-pipeline-fields">
        <legend>Hedef nesne ve Colab ayarları</legend>
        <label>Nesne tarifi (PROMPT)
          <textarea value={prompt} rows={4} required maxLength={2000} placeholder="a red sports car" aria-describedby="text-splat-prompt-help"
            onChange={event => setPrompt(event.target.value)} aria-invalid={Boolean(prompt && !promptValid)} />
        </label>
        <p id="text-splat-prompt-help" className="nb-help">Tek bir nesneyi tarif et; İngilizce tarifler önerilir. En fazla 2000 karakter.</p>
        {prompt && !promptValid && <p className="nb-error" role="alert">Tarif boş olamaz, NUL karakteri içeremez ve 2000 karakteri geçemez.</p>}

        <section className="nb-model-settings" aria-labelledby="text-splat-model-heading">
          <h2 id="text-splat-model-heading">Model ayarları (MODEL SETTINGS)</h2>
          <p className="nb-help">GPU profili modelleri ve kalite bütçelerini birlikte seçer; tüm profiller kararlı TRELLIS yoluyla başlar. TRELLIS.2 deneysel yolunu ayrıca seçebilirsin. Sonrasında her aşamayı değiştirebilirsin. Büyük model tek başına daha iyi 3D sonucu garanti etmez.</p>
          <div className="nb-model-option">
            <label>GPU profili<select value={settings.gpu_preset} onChange={event => chooseGpu(event.target.value)} aria-describedby="text-splat-gpu-help">
              {Object.entries(GPU_PRESETS).map(([value, option]) => <option key={value} value={value}>{option.label}</option>)}
            </select></label>
            <p id="text-splat-gpu-help" className="nb-help">{gpu.description}</p>
          </div>
          <div className="nb-model-option">
            <label>Görsel modeli<select value={settings.image_model} onChange={event => chooseImage(event.target.value)} aria-describedby="text-splat-image-help">
              {Object.entries(IMAGE_MODELS).map(([value, option]) => <option key={value} value={value}>{option.label}</option>)}
            </select></label>
            <p id="text-splat-image-help" className="nb-help">{imageModel.description}</p>
          </div>
          <div className="nb-model-option">
            <label>Arka plan kaldırma<select value={settings.background_model} onChange={event => {
              if (event.target.value in BACKGROUND_MODELS) setSettings(previous => ({ ...previous, background_model: event.target.value as BackgroundModel }));
            }} aria-describedby="text-splat-background-help">
              {Object.entries(BACKGROUND_MODELS).map(([value, option]) => <option key={value} value={value}>{option.label}</option>)}
            </select></label>
            <p id="text-splat-background-help" className="nb-help">{BACKGROUND_MODELS[settings.background_model].description}</p>
          </div>
          <div className="nb-model-option">
            <label>3D üretim yolu<select value={settings.reconstruction_model} onChange={event => {
              if (event.target.value in RECONSTRUCTION_MODELS && event.target.value !== "hunyuan3d") {
                setSettings(previous => ({ ...previous, reconstruction_model: event.target.value as ReconstructionModel }));
              }
            }} aria-describedby="text-splat-route-help text-splat-hunyuan-help">
              {Object.entries(RECONSTRUCTION_MODELS).map(([value, option]) => <option key={value} value={value}
                disabled={value === "hunyuan3d"}>{option.label}</option>)}
            </select></label>
            <p id="text-splat-route-help" className="nb-help">{RECONSTRUCTION_MODELS[settings.reconstruction_model].description}</p>
            <p id="text-splat-hunyuan-help" className="nb-help">Hunyuan3D-2.x — deneysel / sonra: {RECONSTRUCTION_MODELS.hunyuan3d.description}</p>
          </div>
          {defaults && <p className="nb-help" role="status">En az {minimumVram} GiB GPU belleği için ön kontrol uygulanır. Bu proje eşiği ölçülmüş bellek tüketimi veya OOM garantisi değildir. Aşamalar sırayla çalışır; Colab gerçek GPU belleğini indirmelerden önce kontrol eder.</p>}
          {gpuError && <p className="nb-error" role="alert">{gpuError}</p>}
          {blackwellWarning && <p className="nb-help" role="note">Notebook indirilebilir; mevcut kararlı TRELLIS ortamı gerçek RTX PRO 6000 Blackwell GPU’da çalışmaz ve indirmelerden önce durur. Kararlı yol için uyumlu bir GPU oturumu kullan veya deneysel TRELLIS.2 yolunu açıkça seç.</p>}
        </section>

        <div className="nb-inline-fields">
          <div><label>Kalite ayarı<select value={preset} onChange={event => chooseQuality(event.target.value as PipelinePreset)}>
            <option value="baseline">Dengeli</option>
            <option value="quality">Kaliteli</option>
            <option value="ultra">Ultra · en az 38 GiB</option>
          </select></label><p className="nb-help">TRELLIS örnekleme adımlarını ve SDXL seçiliyse görsel adımlarını ayarlar. Aşağıdaki değerleri ayrıca değiştirebilirsin.</p></div>
          <div><label>Tohum (SEED)<input type="number" min="0" max="2147483647" step="1" required value={seed}
            onChange={event => setSeed(event.target.value)} aria-invalid={!seedValid} /></label>
            <p className="nb-help">Görselin rastgelelik tohumu. Aynı ayarlar ve tohum benzer sonuçları tekrar üretmeyi sağlar.</p></div>
        </div>
        {!seedValid && <p className="nb-error" role="alert">Tohum 0–2147483647 arasında bir tam sayı olmalı.</p>}

        <details className="nb-advanced">
          <summary>Aşama kalite ayarları</summary>
          <h3>Görsel üretimi</h3>
          <div className="nb-advanced-grid">
            <NumberSetting name="image_steps" label="Görsel adımları" value={settings.image_steps} min={1} max={imageStepMax} onChange={setNumber}
              help={`1–${imageStepMax} adım. Artırmak süreyi uzatır; kalite artışı garanti değildir. Schnell en fazla 4 adım destekler; klein için 4 adım önerilir.`} />
            <NumberSetting name="image_guidance" label="Görsel guidance / CFG" value={settings.image_guidance} min={0} max={20} fractional onChange={setNumber}
              disabled={settings.image_model === "flux1_schnell" || settings.image_model === "flux2_klein_4b"}
              help="Tarife bağlılığı ayarlar; yüksek değerler doygunluk ve bozulma üretebilir. Qwen’de true CFG; FLUX.1-dev’de negatif tarif varsa true CFG de kullanılır. Schnell 0, klein 1 sabittir." />
            <div><label>Görsel çözünürlüğü<select value={settings.image_resolution} onChange={event => setNumber("image_resolution", Number(event.target.value))}>
              <option value="512">512 × 512</option><option value="768">768 × 768</option><option value="1024">1024 × 1024</option>
            </select></label><p className="nb-help">Büyük görsel daha fazla ayrıntı, süre ve bellek ister; 1024 px önerilir. Çözünürlük 3D splat sayısını belirlemez.</p></div>
          </div>
          <h3>TRELLIS örneklemesi</h3>
          <div className="nb-advanced-grid">
            <NumberSetting name="trellis_seed" label="TRELLIS tohumu" value={settings.trellis_seed} min={0} max={2147483647} onChange={setNumber}
              help="3D üretimin rastgelelik tohumu; görseli değiştirmeden farklı geometri denemeleri sağlar." />
            <NumberSetting name="sparse_steps" label="Sparse-structure adımları" value={settings.sparse_steps} min={1} max={100} onChange={setNumber}
              help="Kaba hacim ve doluluk yapısının örnekleme bütçesi. Fazla adım süreyi artırır; ayrıntı garantisi değildir." />
            <NumberSetting name="sparse_cfg" label="Sparse-structure CFG" value={settings.sparse_cfg} min={0} max={20} fractional onChange={setNumber}
              help="Kaba yapının görsele bağlılığı. Çok yüksek değer geometriyi bozabilir; başlangıç 7.5." />
            <NumberSetting name="slat_steps" label="SLAT adımları" value={settings.slat_steps} min={1} max={100} onChange={setNumber}
              help="TRELLIS’te ince geometri ve görünüş; TRELLIS.2’de şekil/doku örnekleme bütçesi. Fazla adım daha uzun sürer." />
            <NumberSetting name="slat_cfg" label="SLAT CFG" value={settings.slat_cfg} min={0} max={20} fractional onChange={setNumber}
              help={meshRoute ? "TRELLIS.2 şeklinin görsele bağlılığı; yüksek değerler kusurları artırabilir. Doku guidance değeri bu yolda 1 olarak sabittir." : "İnce yapı ve görünüşün görsele bağlılığı. Yüksek değerler kusurları artırabilir; başlangıç 3."} />
          </div>
          {meshRoute && <><h3>Mesh → 3DGS eğitimi</h3><p className="nb-help">Bu sürümde TRELLIS.2 mesh üretimi 512 çözünürlükte, eğitim görüntüleri 512 × 512 px olarak sabittir. Yukarıdaki görsel çözünürlüğü yalnızca ilk metin → görsel aşamasını değiştirir.</p><div className="nb-advanced-grid">
            <NumberSetting name="mesh_views" label="Mesh görüş sayısı" value={settings.mesh_views} min={12} max={120} onChange={setNumber}
              help="12–120 kamera açısından render alır. Daha fazla görüş yüzey kapsamını iyileştirir; hazırlık süresi ve depolama artar." />
            <NumberSetting name="mesh_fit_iterations" label="3DGS fit iterasyonları" value={settings.mesh_fit_iterations} min={100} max={10000} onChange={setNumber}
              help="100–10000 optimizasyon adımı. Daha uzun eğitim mesh görüntülerine daha iyi uyabilir; süre artar, görünmeyen ayrıntı yaratmaz." />
            <NumberSetting name="mesh_splat_cap" label="Mesh splat üst sınırı" value={settings.mesh_splat_cap} min={1000} max={300000} onChange={setNumber}
              help="1000–300000 Gaussian bütçesi. Büyük bütçe ince ayrıntı için alan sağlar; GPU belleği ve PLY boyutu artar. 150000 üzerinde ön kontrol 32 GiB olur." />
          </div></>}
          {!settingsValid && <p className="nb-error" role="alert">Kalite değerlerini belirtilen aralıklarda doldur; adım, görüş, tohum ve splat sayıları tam sayı olmalı.</p>}
        </details>

        <details className="nb-advanced">
          <summary>İsteğe bağlı ayarlar</summary>
          <div className="nb-advanced-grid">
            {imageModel.negative ? <label className="nb-wide-field">İstenmeyen özellikler (NEGATIVE)<textarea value={negative} rows={2} maxLength={2000}
              placeholder="blurry, multiple objects, text, watermark" onChange={event => setNegative(event.target.value)} /></label>
              : <p className="nb-help nb-wide-field">Seçilen görsel modelinde negatif tarif desteklenmez; önceki negatif tarif bu koşuya gönderilmez.</p>}
            <label className="nb-wide-field">Görsel tarzı (STYLE)<input value={style} maxLength={500}
              placeholder="studio product photo, white background" onChange={event => setStyle(event.target.value)} /></label>
            <label className="nb-wide-field">Drive çıktı klasörü (OUTPUT_DIR)<input value={outputDir}
              placeholder="GaussianTests/text_to_splat/&lt;tarif&gt;" aria-invalid={Boolean(output && !output.ok)}
              onChange={event => setOutputDir(event.target.value)} aria-describedby="text-splat-output-help" /></label>
            <p id="text-splat-output-help" className="nb-help nb-wide-field">Boş bırakırsan MyDrive/GaussianTests/text_to_splat/ altında tariften türetilen klasör kullanılır. Özel yol MyDrive’a göre yazılır.</p>
            {output && !output.ok && <p className="nb-error nb-wide-field" role="alert">MyDrive altında göreli bir klasör yaz; yerel bilgisayar yolu, boş yol parçası veya “..” kullanma.</p>}
            {!meshRoute && <><label>Hedef splat sayısı<input type="number" min="1000" max="2000000" step="1" value={targetCount}
              placeholder="Otomatik" aria-invalid={!countValid} onChange={event => setTargetCount(event.target.value)} /></label>
              <p className="nb-help">TRELLIS çıktısı için budama üst sınırı. Boş bırakmak üretilen sayıyı korur; fazlası kayıplı örneklenir, eksik splat eklenmez. İlk morph denemesi için 200000 önerilir.</p></>}
            {!countValid && <p className="nb-error nb-wide-field" role="alert">Hedef splat sayısı 1000–2000000 arasında bir tam sayı olmalı.</p>}
          </div>
        </details>
        <p className="nb-help">Notebook’u Colab’de aç, seçtiğin profile uygun GPU’yu ayarla ve “Tümünü çalıştır” komutunu kullan. Görsel aşaması modern, ayrı bir ortamda çalışır. PLY, önizleme, ara görsel ve model kimliği/sürümü/lisansı içeren manifest Drive’a kaydedilir. HF_TOKEN yalnızca Colab secrets veya ortam değişkeninden okunur.</p>
        <button type="submit" className="nb-generate-button" disabled={!valid || loading}>
          {loading ? "Notebook hazırlanıyor…" : "Metinden splat notebook’unu indir"}
        </button>
      </fieldset>
    </form>
    {error && <p role="alert" className="nb-error">{error}</p>}
    {artifact && <NotebookDownload artifact={artifact} onRegenerate={() => void generate()} onEdit={() => setArtifact(null)} />}
  </div>;
}
