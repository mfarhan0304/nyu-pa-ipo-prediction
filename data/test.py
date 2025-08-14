import math

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def lower_hull(points):
    pts = sorted(set(points))
    hull = []
    for p in pts:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull

def polyline_length(pts):
    return sum(math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]) for i in range(1, len(pts)))

def main():
    n = int(input())
    coords = []
    for _ in range(n):
        x, y = map(int, input().split())
        coords.append((x, y))
    
    hull = lower_hull(coords)
    perimeter = polyline_length(hull)

    print(int(math.floor(perimeter + 0.5)))

if __name__ == "__main__":
    main()