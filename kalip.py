# -*- coding: utf-8 -*-
"""
kalip_araclari.py
------------------
3D STL modelini terzi kalıbı (2D pattern) çıkarmak için çekirdek fonksiyonlar.

Bu modül, orijinal 'terzi_kalip.py' scriptinin mantığını temel alır ama
şunları ekler:
  1. Segmentleri düzleştirirken GERÇEK 3D yüzey alanı ile 2D kalıp alanını
     karşılaştırıp bir "distorsiyon" (gerilme/büzülme) skoru hesaplar.
  2. Bu skoru kullanarak farklı parça sayıları (k) için toplam distorsiyonu
     ölçer ve "dizin noktası" (knee point) yöntemiyle en makul k değerini
     otomatik önerir.
  3. Her adımı (segmentasyon, düzleştirme, alan hesabı) ayrı fonksiyonlara
     böler ki bir arayüz (Streamlit) rahatça çağırabilsin.

NOT: Bu dosya trimesh / scipy / sklearn / shapely kurulu bir ortamda
çalıştırılmalıdır (aynı sizin terzi_kalip.py'yi çalıştırdığınız ortam).
"""

import os
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from shapely.geometry import Polygon

try:
    import trimesh
except ImportError as e:
    raise ImportError(
        "Bu modül 'trimesh' gerektirir. Kurulum: pip install trimesh"
    ) from e


# ---------------------------------------------------------------------------
# 1) MESH OKUMA VE KOMŞULUK MATRİSİ
# ---------------------------------------------------------------------------

def mesh_yukle(dosya_yolu):
    """STL/OBJ/PLY dosyasını yükler ve tek bir Trimesh nesnesi döndürür."""
    mesh = trimesh.load(dosya_yolu, force="mesh")
    return mesh


def komsuluk_matrisi_olustur(mesh):
    """Yüzeyler (face) arası komşuluk ilişkisini seyrek (sparse) matris olarak döndürür."""
    n_faces = len(mesh.faces)
    edges = mesh.face_adjacency
    connectivity = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_faces, n_faces)
    )
    connectivity = (connectivity + connectivity.T) / 2
    return connectivity


# ---------------------------------------------------------------------------
# 2) SEGMENTASYON (parça_sayisi = k)
# ---------------------------------------------------------------------------

def mesh_segmentlere_ayir(mesh, parca_sayisi, connectivity=None, min_oran=0.02):
    """
    Mesh'i normallerine göre 'parca_sayisi' kadar bölgeye ayırır.
    Çöp/küçük kümeleri en yakın komşu büyük kümeye yedirir.

    Döndürür: kume_etiketleri (n_faces,) uzunluğunda etiket dizisi,
              aktif_kumeler (kalan benzersiz etiketler listesi)
    """
    if connectivity is None:
        connectivity = komsuluk_matrisi_olustur(mesh)

    normaller = mesh.face_normals
    n_faces = len(mesh.faces)

    cluster = AgglomerativeClustering(
        n_clusters=parca_sayisi, connectivity=connectivity, linkage="ward"
    )
    kume_etiketleri = cluster.fit_predict(normaller)

    # --- küçük çöp kümeleri eritme ---
    min_yuzey_sayisi = max(1, int(n_faces * min_oran))
    unique, counts = np.unique(kume_etiketleri, return_counts=True)
    kucuk_kumeler = unique[counts < min_yuzey_sayisi]

    if len(kucuk_kumeler) > 0:
        face_adj = mesh.face_adjacency
        for kucuk_id in kucuk_kumeler:
            kucuk_yuzeyler = np.where(kume_etiketleri == kucuk_id)[0]
            mask = np.isin(face_adj[:, 0], kucuk_yuzeyler) | np.isin(
                face_adj[:, 1], kucuk_yuzeyler
            )
            komsu_yuzeyler = np.unique(face_adj[mask].flatten())
            komsu_kumeler = kume_etiketleri[komsu_yuzeyler]
            gecerli_komsular = komsu_kumeler[komsu_kumeler != kucuk_id]
            if len(gecerli_komsular) > 0:
                en_yakin_kume = np.bincount(gecerli_komsular).argmax()
                kume_etiketleri[kucuk_yuzeyler] = en_yakin_kume

    aktif_kumeler = np.unique(kume_etiketleri)
    return kume_etiketleri, aktif_kumeler


def segment_renkleri_uret(kume_etiketleri, aktif_kumeler, seed=42):
    """Her kümeye sabit (tekrarlanabilir) bir RGBA renk atar. face -> renk dizisi döndürür."""
    rng = np.random.RandomState(seed)
    renk_tablosu = rng.randint(0, 255, size=(max(len(aktif_kumeler), 1), 3), dtype=np.uint8)
    yuzey_renkleri = np.zeros((len(kume_etiketleri), 4), dtype=np.uint8)
    for idx, k_id in enumerate(aktif_kumeler):
        renk = renk_tablosu[idx % len(renk_tablosu)]
        yuzey_renkleri[kume_etiketleri == k_id, :3] = renk
        yuzey_renkleri[kume_etiketleri == k_id, 3] = 255
    return yuzey_renkleri, renk_tablosu


# ---------------------------------------------------------------------------
# 3) TEK BİR SEGMENTİ 2D'YE DÜZLEŞTİRME
# ---------------------------------------------------------------------------

def segmenti_duzlestir(submesh, tolerans_orani=0.015, simplify_orani=0.3):
    """
    Bir alt-mesh'i (submesh) en iyi düzleme izdüşürüp pürüzsüz bir 2D
    poligon (shapely Polygon) olarak döndürür.

    Döndürür: final_poligon (shapely Polygon) ya da None (başarısızsa)
    """
    sinir_hatlari = submesh.outline()
    if not sinir_hatlari or len(sinir_hatlari.entities) == 0:
        return None

    merkez = submesh.centroid
    ortalama_normal = submesh.face_normals.mean(axis=0)
    if np.linalg.norm(ortalama_normal) == 0:
        return None
    ortalama_normal /= np.linalg.norm(ortalama_normal)

    donusum_matrisi = trimesh.geometry.plane_transform(merkez, ortalama_normal)

    en_uzun_yol = []
    for path in sinir_hatlari.discrete:
        noktalar_2d_homojen = trimesh.transformations.transform_points(path, donusum_matrisi)
        noktalar_2d = noktalar_2d_homojen[:, :2]
        if len(noktalar_2d) > len(en_uzun_yol):
            en_uzun_yol = noktalar_2d

    if len(en_uzun_yol) < 3:
        return None

    ana_poligon = Polygon(en_uzun_yol)
    if not ana_poligon.is_valid:
        ana_poligon = ana_poligon.buffer(0)
    if ana_poligon.is_empty:
        return None

    boyut_olcegi = submesh.scale
    tolerans = boyut_olcegi * tolerans_orani
    yumusatilmis = ana_poligon.buffer(tolerans, resolution=8).buffer(-tolerans, resolution=8)
    final_poligon = yumusatilmis.simplify(tolerans * simplify_orani, preserve_topology=True)

    if final_poligon.is_empty or not hasattr(final_poligon, "exterior") or final_poligon.exterior is None:
        return None

    return final_poligon


# ---------------------------------------------------------------------------
# 4) TÜM PARÇALARI ÜRETME + ALAN / DİSTORSİYON HESABI
# ---------------------------------------------------------------------------

def parcalari_uret(mesh, kume_etiketleri, aktif_kumeler):
    """
    Her aktif küme için:
      - submesh
      - 2D final poligon (kalıp)
      - gercek_3d_alan (kavisli yüzey alanı, mm^2 ya da modelin birimi neyse)
      - kalip_2d_alan  (düzleştirilmiş poligon alanı)
      - distorsiyon    (|kalip - gercek| / gercek)
    içeren bir liste (dict) döndürür. Düzleşemeyen parçalar atlanır.
    """
    sonuclar = []
    for k_id in aktif_kumeler:
        yuzey_idx = np.where(kume_etiketleri == k_id)[0]
        if len(yuzey_idx) == 0:
            continue
        submesh = mesh.submesh([yuzey_idx], append=True)
        gercek_3d_alan = submesh.area

        poligon = segmenti_duzlestir(submesh)
        if poligon is None:
            continue

        kalip_2d_alan = poligon.area
        distorsiyon = (
            abs(kalip_2d_alan - gercek_3d_alan) / gercek_3d_alan
            if gercek_3d_alan > 0
            else np.nan
        )

        sonuclar.append(
            {
                "kume_id": int(k_id),
                "submesh": submesh,
                "poligon": poligon,
                "gercek_3d_alan": gercek_3d_alan,
                "kalip_2d_alan": kalip_2d_alan,
                "distorsiyon": distorsiyon,
            }
        )
    return sonuclar


def toplam_distorsiyon_skoru(parca_sonuclari):
    """
    Parçaların yüzey alanına göre AĞIRLIKLANDIRILMIŞ ortalama distorsiyonunu
    döndürür (0 = mükemmel düzleşme, 0.10 = %10 alan sapması, vb).
    """
    toplam_alan = sum(p["gercek_3d_alan"] for p in parca_sonuclari)
    if toplam_alan == 0:
        return np.nan
    agirlikli = sum(
        p["gercek_3d_alan"] * p["distorsiyon"]
        for p in parca_sonuclari
        if not np.isnan(p["distorsiyon"])
    )
    return agirlikli / toplam_alan


# ---------------------------------------------------------------------------
# 5) OPTİMAL PARÇA SAYISINI BULMA
# ---------------------------------------------------------------------------

def optimal_parca_sayisi_bul(mesh, k_araligi=range(2, 11), min_oran=0.02, ilerleme_callback=None):
    """
    k_araligi içindeki her k için segmentasyon + düzleştirme + distorsiyon
    skoru hesaplar. Sonuç bir liste olarak döndürülür:
        [{"k": k, "distorsiyon": skor, "parca_sayisi_gercek": len(aktif)}, ...]
    ve ayrıca "kneedle" (diz noktası) yöntemiyle önerilen k değeri.

    ilerleme_callback(k, i, toplam) -> arayüzde ilerleme çubuğu için opsiyonel.
    """
    connectivity = komsuluk_matrisi_olustur(mesh)
    sonuc_listesi = []
    k_liste = list(k_araligi)

    for i, k in enumerate(k_liste):
        etiketler, aktif = mesh_segmentlere_ayir(mesh, k, connectivity=connectivity, min_oran=min_oran)
        parcalar = parcalari_uret(mesh, etiketler, aktif)
        skor = toplam_distorsiyon_skoru(parcalar)
        sonuc_listesi.append(
            {"k": k, "distorsiyon": skor, "parca_sayisi_gercek": len(parcalar)}
        )
        if ilerleme_callback:
            ilerleme_callback(k, i + 1, len(k_liste))

    onerilen_k = _kneedle_nokta_bul(sonuc_listesi)
    return sonuc_listesi, onerilen_k


def _kneedle_nokta_bul(sonuc_listesi):
    """
    Basit bir 'diz noktası' (knee point) tespiti:
    Distorsiyon eğrisi genelde k arttıkça azalır (azalan getiri).
    İlk ve son noktayı birleştiren doğruya en uzak nokta = diz noktası.
    """
    gecerli = [s for s in sonuc_listesi if not np.isnan(s["distorsiyon"])]
    if len(gecerli) < 3:
        return gecerli[0]["k"] if gecerli else None

    ks = np.array([s["k"] for s in gecerli], dtype=float)
    ys = np.array([s["distorsiyon"] for s in gecerli], dtype=float)

    # 0-1 aralığına normalize et
    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-9)
    ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)

    x1, y1 = ks_n[0], ys_n[0]
    x2, y2 = ks_n[-1], ys_n[-1]
    # noktanın (x1,y1)-(x2,y2) doğrusuna dik uzaklığı
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy) + 1e-9
    uzakliklar = np.abs(dy * (ks_n - x1) - dx * (ys_n - y1)) / norm

    en_uzak_idx = np.argmax(uzakliklar)
    return int(ks[en_uzak_idx])
def komsu_bilgisi_hesapla(mesh, kume_etiketleri):
    """Her yüzeyin komşularına bakarak hangi kümenin hangi kümeye değdiğini bulur."""
    face_adj = mesh.face_adjacency
    komsuluklar = {} # (parca_i, parca_j) -> [kenar_noktalari]
    
    for i, j in face_adj:
        etiket_i = kume_etiketleri[i]
        etiket_j = kume_etiketleri[j]
        
        if etiket_i != etiket_j:
            # Bu iki yüzey farklı segmentlere ait, yani burası bir dikiş hattı
            seam = tuple(sorted((etiket_i, etiket_j)))
            if seam not in komsuluklar:
                komsuluklar[seam] = []
            
            # Ortak kenar noktalarını bul
            ortak_vertex_id = np.intersect1d(mesh.faces[i], mesh.faces[j])
            komsuluklar[seam].append(mesh.vertices[ortak_vertex_id])
            
    return komsuluklar