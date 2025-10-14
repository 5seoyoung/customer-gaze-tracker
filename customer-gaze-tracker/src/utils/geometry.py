from shapely.geometry import Point, Polygon

def point_in_which_roi(pt_xy, roi_cfg):
    px, py = pt_xy
    for r in roi_cfg["rois"]:
        if Polygon(r["polygon"]).contains(Point(px, py)):
            return r["name"]
    return None

def gaze_intersection_2d(face_box, gaze_vec, scale=150):
    x1,y1,x2,y2 = face_box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    dx, dy = float(gaze_vec[0]), float(gaze_vec[1])
    end = (int(cx + dx*scale), int(cy + dy*scale))
    return (int(cx), int(cy)), end
