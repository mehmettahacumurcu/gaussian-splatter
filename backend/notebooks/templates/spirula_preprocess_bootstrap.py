# Embedded after the shared, pinned Spirula installer constants and log helpers.
from google.colab import drive
drive.mount('/content/drive')

RUN_ID = CONFIG['dataset_name'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
WORK = Path('/content/spirula_preprocessing') / RUN_ID
DRIVE_OUT = Path('/content/drive/MyDrive') / CONFIG['output_path'] / RUN_ID
LOGS = WORK / 'logs'
OUT = WORK / 'outputs'
MANIFEST_PATH = WORK / 'manifest.json'
BINARY = WORK / 'spirula'
LOGS.mkdir(parents=True, exist_ok=False)
OUT.mkdir(parents=True)
manifest = {'run_id': RUN_ID, 'source_commit': COMMIT, 'settings': CONFIG,
            'status': 'starting', 'drive_export_status': 'pending'}


def retry_drive_export():
    errors = []
    archive = OUT / 'dataset.zip'
    if archive.is_file():
        ok, error = _copy_file_to_drive(archive, DRIVE_OUT / 'dataset.zip', 'dataset')
        if not ok:
            errors.append(error)
    for relative in ('logs', 'reports'):
        ok, error = _copy_tree_to_drive(WORK / relative, DRIVE_OUT / relative, relative)
        if not ok:
            errors.append(error)
    manifest['drive_export_status'] = 'pending' if errors else 'complete'
    if manifest.get('preprocessing_status') == 'complete':
        manifest['status'] = 'export_pending' if errors else 'complete'
    manifest['drive_export_errors'] = errors
    save_manifest()
    ok, error = _copy_file_to_drive(MANIFEST_PATH, DRIVE_OUT / 'manifest.json', 'manifest')
    if not ok:
        errors.append(error)
        manifest['drive_export_status'] = 'pending'
        manifest['drive_export_errors'] = errors
        if manifest.get('preprocessing_status') == 'complete':
            manifest['status'] = 'export_pending'
        save_manifest()
    print('Yerel dosyalar:', WORK)
    if errors:
        print('Drive aktarımı bekliyor; oturumu kapatmadan bağlantıyı düzeltip bu hücreyi tekrar çalıştır.')
    elif archive.is_file():
        print('Dataset ZIP:', DRIVE_OUT / 'dataset.zip')
        print('Rapor ve ayarlar:', DRIVE_OUT / 'reports')
    return errors


save_manifest()
for entry in CONFIG['inputs']:
    source = Path('/content/drive/MyDrive') / entry['path']
    require(source.is_file() if entry['kind'] == 'video' else source.is_dir(),
            f'Giriş bulunamadı: {source}. Drive hesabını ve yolu kontrol et.', 'input')
print('Yerel çalışma:', WORK)
print('Drive çıktısı:', DRIVE_OUT)
