"""Spirula dataset preparation contract, UI catalog and portable notebook builder."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Literal

import nbformat
from pydantic import Field, field_validator, model_validator

from .drive_paths import normalize_input_folder
from .models import StrictModel
from .source import NotebookSource

TEMPLATES = Path(__file__).parent / 'templates'
GROUPS = ['Ayarlar', 'Maskeler', 'Geometri', 'Gelişmiş', 'Sensörler']


def setting(default, title, section='Ayarlar', description='', when=None, **constraints):
    return Field(default=default, title=title, description=description,
                 json_schema_extra={'section': section, 'when': when}, **constraints)


class PreprocessSettings(StrictModel):
    quality: Literal['low', 'medium', 'high', 'extreme'] = setting('high', 'Kalite')
    camera_model: Literal['simple-pinhole', 'pinhole', 'radial', 'opencv', 'full-opencv', 'opencv-fisheye', 'thin-prism-fisheye', 'equirectangular'] = setting('opencv', 'Kamera / objektif')
    camera_mode: Literal['single', 'folder', 'image'] = setting('folder', 'Kamera paylaşımı', description='single: tek kamera · folder: klasör başına · image: görüntü başına')
    pairs: Literal['auto', 'exhaustive', 'sequential', 'prefilter'] = setting('auto', 'Görüntü eşleştirme')
    adaptive_fps: bool = setting(False, 'Hızlı harekete göre ayarla', description='Videoda FPS hedefini hareket yoğunluğuna göre dağıtır.')
    sharpness_window: int = setting(3, 'Keskinlik penceresi', description='Video kareleri arasından en keskinini seçer. 1: kapalı.', ge=1, le=120)
    mask_objects: bool = setting(False, 'Hareketli veya istenmeyen nesneleri kaldır', 'Maskeler')
    mask_model: Literal['sam3-q4_0', 'sam3-f16'] = setting('sam3-q4_0', 'Maskeleme modeli', 'Maskeler', 'SAM 3 metin istemini kullanır; seçilirse model indirilir.', when='mask_objects')
    mask_prompt: str = setting('person', 'Kaldırılacak nesneler', 'Maskeler', 'İngilizce kavramları noktalı virgülle ayır: person; car', when='mask_objects', max_length=1000)
    mask_negative_prompt: str = setting('', 'Korunacak nesneler', 'Maskeler', when='mask_objects', max_length=1000)
    mask_threshold: float = setting(0.5, 'Maske algılama eşiği', 'Maskeler', when='mask_objects', ge=0, le=1)
    mask_dilate: float = setting(0.05, 'Maske kenar payı', 'Maskeler', when='mask_objects', ge=-0.5, le=0.5)
    mask_max_size: int = setting(1600, 'Maskeleme en uzun kenarı', 'Maskeler', when='mask_objects', ge=256, le=8192)
    static_mask: bool = setting(False, 'Karenin sabit alanlarını çıkar', 'Maskeler', 'Sabit lens çerçevesini bulur; aşağıda şekil de tanımlanabilir.')
    static_shapes: str = setting('', 'Sabit maske şekilleri', 'Maskeler', 'Boş: otomatik kenar. Örnek: rect 0,0,1,0.95 (normalize koordinatlar).', when='static_mask', max_length=2000)
    generate_geometry: bool = setting(True, 'Derinlik ve normalleri kestir', 'Geometri')
    geometry_model: Literal['moge2-vits', 'moge2-vitb', 'moge2-vitl', 'metric3d-vit-small', 'metric3d-vit-large', 'metric3d-vit-giant2'] = setting('moge2-vitb', 'Geometri modeli', 'Geometri', when='generate_geometry')
    generate_depth: bool = setting(True, 'Normal haritalarına derinliği de ekle', 'Geometri', when='generate_geometry')
    geometry_max_size: int = setting(1064, 'Geometri en uzun kenarı', 'Geometri', when='generate_geometry', ge=256, le=8192)
    geometry_tokens: int = setting(3600, 'MoGe ayrıntı bütçesi (token)', 'Geometri', 'MoGe için 1200–3600; Metric3D bu ayarı kullanmaz.', when='generate_geometry', ge=1200, le=3600)
    capture_kind: Literal['video', 'individual', 'internet'] = setting('video', 'Çekim türü', 'Gelişmiş', 'Tek tek çekilmiş fotoğraflar için individual seç.')
    overlap: int = setting(10, 'Sıralı örtüşme', 'Gelişmiş', ge=1, le=1000)
    loop_closure: bool = setting(True, 'Döngü kapatma', 'Gelişmiş')
    prefilter_sequential: bool = setting(True, 'Çift seçiminde sıralı komşuları da eşleştir', 'Gelişmiş')
    focal: float = setting(0, 'Başlangıç odak uzaklığı (px)', 'Gelişmiş', '0: otomatik.', ge=0, le=1000000)
    distortion: str = setting('', 'Başlangıç bozulması (k1,k2,...)', 'Gelişmiş', max_length=300)
    distortion_refine: Literal['mapping', 'final', 'fixed'] = setting('mapping', 'Bozulmanın iyileştirilmesi', 'Gelişmiş')
    per_image_intrinsics: bool = setting(False, 'Sonda görüntü başına iç parametreler', 'Gelişmiş')
    final_free_rig: bool = setting(False, 'Sonunda rigi serbest bırak', 'Gelişmiş')
    max_features: int = setting(0, 'Görüntü başına en çok öznitelik', 'Gelişmiş', '0: kalite ön ayarı.', ge=0, le=1000000)
    max_image_size: int = setting(0, 'SfM en büyük görüntü boyutu', 'Gelişmiş', '0: kalite ön ayarı. Kaydedilen kaynak görüntüler küçültülmez.', ge=0, le=32768)
    keep_intermediates: bool = setting(False, 'Ara dosyaları arşive ekle', 'Gelişmiş', 'features/ ve matches.bin ZIP boyutunu artırabilir. Yerel çalışma dosyaları her durumda korunur.')
    sensor_gauge: Literal['auto', 'up', 'none'] = setting('auto', "Videonun IMU ve GPS'i", 'Sensörler', 'auto: yukarı yönü ve metrik ölçek · up: yalnız yukarı · none: kapalı. Yalnız dosyada sensör verisi varsa etkilidir.')
    metric_gps: Literal['none', 'horizontal', 'full'] = setting('horizontal', "Fotoğrafların GPS'i", 'Sensörler')
    exif_attitude: Literal['auto', 'up', 'none'] = setting('auto', 'Fotoğrafların duruşu', 'Sensörler')

    @field_validator('distortion')
    @classmethod
    def distortion_numbers(cls, value):
        if value and not re.fullmatch(r'\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*,\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)*\s*', value):
            raise ValueError('Bozulma katsayıları virgülle ayrılmış sayılar olmalı.')
        return value.strip()

    @field_validator('static_shapes')
    @classmethod
    def shapes_only(cls, value):
        # File paths and arbitrary CLI extensions are intentionally not a shape.
        for shape in filter(str.strip, value.split(';')):
            if not re.fullmatch(r'\s*-?(?:rect|ellipse)\s+[0-9.+,\s-]+', shape):
                raise ValueError('Sabit maske için rect veya ellipse koordinatları kullan.')
            numbers = shape.strip().split(None, 1)[1].split(',')
            if len(numbers) != 4 or any(not 0 <= float(n) <= 1 for n in numbers):
                raise ValueError('Şekil dört adet 0–1 koordinatı içermeli.')
        return value.strip()

    @model_validator(mode='after')
    def mask_needs_prompt(self):
        if self.mask_objects and not self.mask_prompt.strip():
            raise ValueError('Maskeleme için kaldırılacak nesneleri yaz.')
        return self


class PreprocessInput(StrictModel):
    kind: Literal['video', 'photos'] = 'video'
    path: str
    fps: float = Field(default=4, gt=0, le=120, allow_inf_nan=False)

    @field_validator('path')
    @classmethod
    def drive_path(cls, value):
        return normalize_input_folder(value)


class PreprocessSpec(StrictModel):
    schema_version: Literal[1] = 1
    inputs: list[PreprocessInput] = Field(min_length=1, max_length=32)
    output_path: str = 'GaussianTests/datasets'
    dataset_name: str = Field(default='spirula_dataset', pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,59}$')
    settings: PreprocessSettings = Field(default_factory=PreprocessSettings)

    @field_validator('output_path')
    @classmethod
    def output_drive_path(cls, value):
        return normalize_input_folder(value)


def preprocess_catalog():
    schema = PreprocessSettings.model_json_schema()
    return {'schema_version': 1, 'spirula_version': '2026.9.24', 'groups': GROUPS,
            'fields': schema['properties'], 'defaults': PreprocessSettings().model_dump()}


def _template_cell(marker):
    nb = nbformat.read(TEMPLATES / 'spirula.ipynb', 4)
    cells = [c.source for c in nb.cells if c.cell_type == 'code' and marker in c.source]
    if len(cells) != 1:
        raise ValueError(f'Spirula runtime template drift: {marker}')
    return cells[0]


def _shared_bootstrap():
    """Reuse installer pins and logging functions, without importing training setup."""
    original = ast.parse(_template_cell('def run_logged('))
    constants = {'RELEASE_TAG', 'VERSION', 'COMMIT', 'ASSET_NAME', 'ASSET_URL', 'ASSET_SIZE', 'EXPECTED_SHA256', 'RELEASE_API'}
    functions = {'save_manifest', '_copy_file_to_drive', '_copy_tree_to_drive', 'fail', 'require', 'run_logged'}
    picked = []
    for node in original.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not (isinstance(node, ast.ImportFrom) and node.module == 'google.colab'):
            picked.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id in constants for n in node.targets):
            picked.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            picked.append(node)
    found = {n.name for n in picked if isinstance(n, ast.FunctionDef)}
    if found != functions:
        raise ValueError('Spirula logging template drift')
    text = ast.unparse(ast.Module(body=picked, type_ignores=[]))
    # Record tolerated partial/geometry exit codes too; no failure is silently called success.
    marker = 'if proc.returncode not in accepted_exit_codes:'
    if text.count(marker) != 1:
        raise ValueError('Spirula process runner drift')
    text = text.replace(marker, "log.write(f'\\nEXIT_CODE: {proc.returncode}\\n')\n            " + marker)
    return text


def build_preprocess_notebook(spec: PreprocessSpec, *, source: NotebookSource):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    bootstrap = _shared_bootstrap() + '\n\n' + (TEMPLATES / 'spirula_preprocess_bootstrap.py').read_text(encoding='utf8')
    installer = _template_cell('def select_release_asset(')
    # The official binary download/dependency checks are shared, training help is not.
    marker = "help_output = run_logged('train_help'"
    if installer.count(marker) != 1:
        raise ValueError('Spirula installer template drift')
    installer = installer.split(marker)[0]
    installer = installer.replace("'vulkan-tools'", "'vulkan-tools', 'ffmpeg'")
    installer += "\nimport sys\nrun_logged('python_dependencies', [sys.executable, '-c', 'import cv2, numpy, PIL'], timeout=60)\n"
    runtime = (TEMPLATES / 'spirula_preprocess_runtime.py').read_text(encoding='utf8')
    nb = nbformat.v4.new_notebook(cells=[
        md('# Spirula · Dataset hazırlama\n\nVideo ve fotoğraflardan **yalnız preprocessing**. GPU seç, Drive hesabını bağla ve hücreleri sırayla çalıştır. '
           'Eğitim ayrı notebook’ta yapılır. Kaynaklar yerel diske kopyalanır; tüm SfM bileşenleri saklanır. '
           'Geometri haritaları en büyük bileşen için üretilir.\n\n[Spirula Studio](https://github.com/harry7557558/spirula-studio) '
           'v2026.9.24 resmi ikilisi kullanılır. SAM 3 seçilirse [model lisansı](https://github.com/facebookresearch/sam3/blob/main/LICENSE) geçerlidir.'),
        md('## 1. Uygulamada seçilen ayarlar\nYollar MyDrive’a göredir. Her çalıştırma yeni bir çıktı klasörü oluşturur.'),
        code('CONFIG = ' + repr(spec.model_dump()), metadata={'stage': 'config'}),
        md('## 2. Drive ve yerel günlükler'), code(bootstrap),
        md('## 3. Çalışma zamanı ve sabitlenmiş Spirula sürümü'),
        code(_template_cell('libc_name, libc_version =')), code(installer),
        md('## 4. NVIDIA Vulkan kontrolü'), code(_template_cell('def recover_matching_nvidia_userspace(')),
        md('## 5. Dataset hazırlama araçları'), code(runtime, metadata={'stage': 'recipe'}),
        code("prep = Preprocessor(CONFIG, WORK, BINARY, GPU_SELECTOR, run_logged)\nverify_cli(prep)\nmanifest['settings'] = CONFIG\nsave_manifest()"),
        md('## 6. Kareler ve maskeler\nVideoda seçilen FPS kaynak kare hızına göre yuvarlanır; gerçek değer rapora yazılır. Fotoğraflar yeniden örneklenmez.'),
        code("prep.stage('prepare')"),
        md('## 7. Kamera eşleştirme ve kalite raporu\nKısmi sonuç (çıkış 3) doğrulanıp korunur. Kapsamı ve bileşenleri incele; daha çok GPU eksik görüntüleri tamamlamaz.'),
        code("prep.stage('reconstruct')\nprint(json.dumps(prep.report, indent=2, ensure_ascii=False))"),
        md('## 8. Normal ve derinlik haritaları\nKapalıysa atlanır. Açıkken seçilen en büyük bileşenin tüm haritaları okunup doğrulanır.'),
        code("prep.stage('geometry')"),
        md('## 9. Dataset ZIP ve Drive çıktısı'),
        code("archive = prep.stage('package')\nmanifest['preprocessing_status'] = 'complete'\nmanifest['archive'] = str(archive)\nmanifest['sfm_report'] = prep.report\nsave_manifest()\nretry_drive_export()"),
        md('## Drive aktarımını yeniden dene\nDrive bağlantısı kesilirse oturumu kapatmadan yeniden bağlayıp bu hücreyi çalıştır. Preprocessing tekrarlanmaz. Hata sonrasında da yerel dosyalar korunur.'),
        code('retry_drive_export()'),
        md('## Sonraki adım\nYazdırılan **dataset.zip** yolunu uygulamada “Spirula preprocessing + Spirula trainer → Dataset ZIP” girişine ver. '
           'Bizim trainer ile denemek için “Spirula dataset + our trainer” da kullanılabilir; bu ekrandaki modellerden mevcut importer yalnız tek kamera ve simple-pinhole, pinhole veya opencv destekler. '
           'Geometri yalnız rapordaki `selected_model` içindir; diğer `sparse/N` bileşenleri de ZIP’te korunur. '
           'SfM kayıt kapsamı ve seçili bileşenin kapsamı ayrı değerlerdir. Hazır dataset ile eğitimde yeniden SfM gerekmez.'),
    ], metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                 'pipeline': {'id': 'spirula_preprocess', 'template_version': 1, 'spec': spec.model_dump(), 'generator_commit': source.commit_sha}})
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    nbformat.validate(nb)
    return nb


def preprocess_filename(spec):
    return f'{spec.dataset_name}_preprocessing.ipynb'
