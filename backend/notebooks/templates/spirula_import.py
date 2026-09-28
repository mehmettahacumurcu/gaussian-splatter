"""Spirula single-camera COLMAP BIN importer for the experimental notebook."""
from pathlib import Path
import struct, time, json
import cv2
import numpy as np

def read(f, fmt):
    n = struct.calcsize('<' + fmt)
    data = f.read(n)
    if len(data) != n:
        raise ValueError('Eksik/bozuk COLMAP dosyası')
    return struct.unpack('<' + fmt, data)

def convert_dataset(source, output, model_name='0', long_edge=1920):
    source, output = Path(source), Path(output)
    model_src = source / 'sparse' / model_name
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        if not (model_src / name).is_file():
            raise ValueError(f'Eksik: {model_src / name}')
    if output.exists():
        raise ValueError('Çıktı klasörü zaten var; yeni bir çalışma klasörü kullanın.')
    started = time.perf_counter()
    with (model_src / 'cameras.bin').open('rb') as f:
        if read(f, 'Q')[0] != 1:
            raise ValueError('Bu test tek kamera kalibrasyonu destekler. Birden fazla lens/video için ayrı adaptör gerekir.')
        cid, typ, w, h = read(f, 'iiQQ')
        counts = {0: 3, 1: 4, 2: 4, 4: 8}
        if typ not in counts:
            raise ValueError(f'Desteklenmeyen kamera modeli {typ}; 360/fisheye bu test kapsamında değil.')
        pars = read(f, 'd' * counts[typ])
    if typ in (0, 2):
        fx = fy = pars[0]; cx, cy = pars[1:3]
        dist = np.array([pars[3] if typ == 2 else 0., 0., 0., 0.])
    else:
        fx, fy, cx, cy = pars[:4]
        dist = np.array(pars[4:] if typ == 4 else [0., 0., 0., 0.])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64)
    if not np.isfinite(K).all() or not np.isfinite(dist).all() or min(fx, fy, w, h) <= 0:
        raise ValueError('Geçersiz kamera parametreleri')
    scale = min(1., long_edge / max(w, h))
    ow, oh = max(1, round(w * scale)), max(1, round(h * scale))
    if np.any(dist):
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (ow, oh))
    else:
        newK = K.copy(); newK[0] *= ow / w; newK[1] *= oh / h
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, newK, (ow, oh), cv2.CV_32FC1)
    rows = []
    with (model_src / 'images.bin').open('rb') as f:
        for _ in range(read(f, 'Q')[0]):
            header = read(f, 'idddddddi')
            name = bytearray()
            while True:
                b = f.read(1)
                if not b: raise ValueError('Bozuk görüntü adı')
                if b == b'\0': break
                name.extend(b)
            n = read(f, 'Q')[0]
            data = f.read(n * 24)
            if len(data) != n * 24: raise ValueError('Bozuk 2D gözlemler')
            pts = np.frombuffer(data, dtype=[('x', '<f8'), ('y', '<f8'), ('id', '<i8')])
            name = name.decode('utf-8')
            path = (source / 'images' / name).resolve()
            if not path.is_relative_to((source / 'images').resolve()) or not path.is_file():
                raise ValueError(f'Geçersiz/eksik görüntü: {name}')
            if header[-1] != cid: raise ValueError('Kamera ID uyumsuzluğu')
            rows.append((name, header, pts, path))
    if len(rows) < 10: raise ValueError('En az 10 hizalanmış görüntü gerekli')
    if len({r[0] for r in rows}) != len(rows): raise ValueError('Tekrarlanan görüntü adı')
    model = output / 'colmap/sparse/0'; frames = output / 'frames'
    model.mkdir(parents=True); frames.mkdir()
    (model / 'cameras.txt').write_text(f'# Number of cameras: 1\n{cid} PINHOLE {ow} {oh} {newK[0,0]} {newK[1,1]} {newK[0,2]} {newK[1,2]}\n')
    mapping = {}
    with (model / 'images.txt').open('w', encoding='utf-8') as f:
        f.write(f'# Number of images: {len(rows)}\n')
        for i, (name, header, pts, path) in enumerate(sorted(rows)):
            dest = f'frame_{i:06d}.png'; mapping[name] = dest
            im = cv2.imread(str(path))
            if im is None or im.shape[:2] != (h, w): raise ValueError(f'Görüntü boyutu hatalı: {name}')
            converted = cv2.remap(im, mx, my, cv2.INTER_LINEAR)
            if not cv2.imwrite(str(frames / dest), converted, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f'Yazılamadı: {dest}')
            xy = np.column_stack((pts['x'], pts['y'])).reshape(-1, 1, 2)
            uv = cv2.undistortPoints(xy, K, dist, P=newK).reshape(-1, 2) if len(pts) else np.empty((0,2))
            f.write(' '.join(map(str, header)) + ' ' + dest + '\n')
            f.write(' '.join(f'{x:.12g} {y:.12g} {int(pid)}' for (x,y),pid in zip(uv,pts['id'])) + '\n')
            if (i+1) % 100 == 0: print(f'{i+1}/{len(rows)} görüntü hazır', flush=True)
    with (model_src / 'points3D.bin').open('rb') as f, (model / 'points3D.txt').open('w') as g:
        npoints = read(f, 'Q')[0]
        if not npoints: raise ValueError('Başlangıç noktası bulunamadı')
        g.write(f'# Number of points: {npoints}\n')
        for _ in range(npoints):
            head = read(f, 'QdddBBBd'); n = read(f, 'Q')[0]; track = read(f, 'ii' * n)
            g.write(' '.join(map(str, head + track)) + '\n')
    report = dict(images=len(rows), points=npoints, width=ow, height=oh, seconds=time.perf_counter()-started,
                  source_model=model_name, source_camera_model=typ, mapping=mapping,
                  notes=['Kamera pozları ve 3D noktalar korundu; görüntü ve 2D gözlemler birlikte düzeltildi.',
                         'alpha=0 lens düzeltmesi görüş alanını bir miktar daraltabilir.',
                         'points3D hata değerleri özgün modelden gelir; yeniden hesaplanmadı.',
                         'Normaller/derinlik/maskeler bu testte kullanılmaz.'])
    (output / 'import_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report
