# -*- coding: utf-8 -*-
"""
kalip_araclari.py
------------------
3D STL modelini terzi kalıbı (2D pattern) çıkarmak için çekirdek fonksiyonlar.
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
# 1) MESH OKUMA, DİNAMİK ÖLÇEKLENDİRME VE KOMŞULUK MATRİSİ
# ---------------------------------------------------------------------------

def mesh_yukle(dosya_yolu):
    """STL/OBJ/PLY dosyasını yükler ve tek bir Trimesh nesnesi döndürür."""
    mesh = trimesh.load(dosya_yolu, force="mesh")
    return mesh


def mesh_olceklendir(mesh, hedef_cap_cm=22.0):
    """
    Modelin alt kısmındaki boşluğun (taban) genişliğini hesaplar ve 
    istenen hedef çapa ulaşması için modeli dinamik olarak ölçeklendirir.
    """
    noktalar = mesh.vertices
    
    # Modelin bounding box (sınırlayıcı kutu) limitleri
    min_koordinatlar = np.min(noktalar, axis=0)
    max_koordinatlar = np.max(noktalar, axis=0)
    
    # Y ekseninde tabana en yakın %5'lik dilimi alt kesit olarak alıyoruz
    y_esik_degeri = min_koordinatlar[1] + (max_koordinatlar[1] - min_koordinatlar[1]) * 0.05
    taban_noktalari = noktalar[noktalar[:, 1] <= y_esik_degeri]
    
    if len(taban_noktalari) == 0:
        # Taban bulunamazsa orijinal mesh'i değişmeden dön
        return mesh, 1.0, 0.0
        
    # X ve Z eksenlerindeki açıklığı bul
    taban_x_uzunlugu = np.max(taban_noktalari[:, 0]) - np.min(taban_noktalari[:, 0])
    taban_z_uzunlugu = np.max(taban_noktalari[:, 2]) - np.min(taban_noktalari[:, 2])
    
    # Mevcut taban çapını en geniş eksen olarak belirliyoruz
    mevcut_cap = max(taban_x_uzunlugu, taban_z_uzunlugu)
    
    # İhtiyacımız olan scale (ölçek) katsayisini hesapla
    olcek_katsayisi = hedef_cap_cm / (mevcut_cap + 1e-9) # Sıfıra bölünme hatasını önlemek için küçük bir değer ekledik
    
    # Trimesh'in yerleşik metodu ile modelin tüm noktalarını tam oranda büyüt/küçült
    mesh.apply_scale(olcek_katsayisi)
    
    return mesh, olcek_katsayisi, mevcut_cap


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
    poligon (shapely Polygon) ve dönüşüm matrisi olarak döndürür.
    """
    sinir_hatlari = submesh.outline()
    if not sinir_hatlari or len(sinir_hatlari.entities) == 0:
        return None, None

    merkez = submesh.centroid
    ortalama_normal = submesh.face_normals.mean(axis=0)
    if np.linalg.norm(ortalama_normal) == 0:
        return None, None
    ortalama_normal /= np.linalg.norm(ortalama_normal)

    donusum_matrisi = trimesh.geometry.plane_transform(merkez, ortalama_normal)

    en_uzun_yol = []
    for path in sinir_hatlari.discrete:
        noktalar_2d_homojen = trimesh.transformations.transform_points(path, donusum_matrisi)
        noktalar_2d = noktalar_2d_homojen[:, :2]
        if len(noktalar_2d) > len(en_uzun_yol):
            en_uzun_yol = noktalar_2d

    if len(en_uzun_yol) < 3:
        return None, None

    ana_poligon = Polygon(en_uzun_yol)
    if not ana_poligon.is_valid:
        ana_poligon = ana_poligon.buffer(0)
    if ana_poligon.is_empty:
        return None, None

    boyut_olcegi = submesh.scale
    tolerans = boyut_olcegi * tolerans_orani
    yumusatilmis = ana_poligon.buffer(tolerans, resolution=8).buffer(-tolerans, resolution=8)
    final_poligon = yumusatilmis.simplify(tolerans * simplify_orani, preserve_topology=True)

    if final_poligon.is_empty or not hasattr(final_poligon, "exterior") or final_poligon.exterior is None:
        return None, None

    return final_poligon, donusum_matrisi


def parcalari_uret(mesh, kume_etiketleri, aktif_kumeler):
    """Her aktif küme için 2D kalıp, alan, distorsiyon ve uzay dönüşüm matrisi üretir."""
    sonuclar = []
    for k_id in aktif_kumeler:
        yuzey_idx = np.where(kume_etiketleri == k_id)[0]
        if len(yuzey_idx) == 0:
            continue
        submesh = mesh.submesh([yuzey_idx], append=True)
        gercek_3d_alan = submesh.area

        sonuc = segmenti_duzlestir(submesh)
        if sonuc is None or sonuc[0] is None:
            continue
            
        poligon, don_mat = sonuc
        kalip_2d_alan = poligon.area
        
        distorsiyon = (
            abs(kalip_2d_alan - gercek_3d_alan) / gercek_3d_alan
            if gercek_3d_alan > 0 else np.nan
        )

        sonuclar.append({
            "kume_id": int(k_id),
            "submesh": submesh,
            "poligon": poligon,
            "gercek_3d_alan": gercek_3d_alan,
            "kalip_2d_alan": kalip_2d_alan,
            "distorsiyon": distorsiyon,
            "donusum_matrisi": don_mat,
        })
    return sonuclar

def toplam_distorsiyon_skoru(parca_sonuclari):
    """
    Parçaların yüzey alanına göre ağırlıklı ortalama distorsiyonunu hesaplar.
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
    Tavsiye edilen optimal k değerini hesaplar.
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
    gecerli = [s for s in sonuc_listesi if not np.isnan(s["distorsiyon"])]
    if len(gecerli) < 3:
        return gecerli[0]["k"] if gecerli else None

    ks = np.array([s["k"] for s in gecerli], dtype=float)
    ys = np.array([s["distorsiyon"] for s in gecerli], dtype=float)

    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-9)
    ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)

    x1, y1 = ks_n[0], ys_n[0]
    x2, y2 = ks_n[-1], ys_n[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy) + 1e-9
    uzakliklar = np.abs(dy * (ks_n - x1) - dx * (ys_n - y1)) / norm

    en_uzak_idx = np.argmax(uzakliklar)
    return int(ks[en_uzak_idx])


# --- SEVİYE 2: MATEMATİKSEL OLARAK DOĞRU DİKİŞ EŞLEŞTİRME ---
    if dikis_merkezleri is not None and "donusum_matrisi" in p:
            kume_id = p["kume_id"]
            don_mat = p["donusum_matrisi"]
            
            for seam, merkez_3d in dikis_merkezleri.items():
                if kume_id in seam:
                    hedef_kume = seam[0] if seam[1] == kume_id else seam[1]
                    hedef_parca_no = kume_to_parca.get(hedef_kume, "?")
                    
                    # 1. 3D merkezi dönüşüm matrisi ile 2D düzlemine taşı
                    pt_3d = np.array([merkez_3d[0], merkez_3d[1], merkez_3d[2], 1.0])
                    pt_2d_hom = np.dot(don_mat, pt_3d)
                    pt_2d = pt_2d_hom[:2]
                    
                    # 2. Noktayı dikiş hattına (dis_cevre) sabitle
                    pnt = Point(pt_2d)
                    projected_dist = dis_cevre.project(pnt)
                    snap_pt = dis_cevre.interpolate(projected_dist)
                    
                    # 3. Harf atama (her dikiş hattı için benzersiz bir harf)
                    seam_keys = list(dikis_merkezleri.keys())
                    harf = chr(65 + (seam_keys.index(seam) % 26))
                    
                    # 4. Görselleştirme (zorder ile en üste al)
                    ax.plot(snap_pt.x, snap_pt.y, marker='s', color='gold', markersize=12, markeredgecolor='black', zorder=5)
                    ax.text(snap_pt.x, snap_pt.y, f"{harf}\n(P{hedef_parca_no})", 
                            color="black", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6)
