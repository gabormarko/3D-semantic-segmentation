import sys
from plyfile import PlyData

if len(sys.argv) < 2:
    print("Usage: python check_ply_fields.py <input.ply>")
    sys.exit(1)

ply = PlyData.read(sys.argv[1])
vertex = ply['vertex']
print("Available fields in .ply vertex:")
print(vertex.data.dtype)
print("First 5 rows:")
print(vertex.data[:5])
