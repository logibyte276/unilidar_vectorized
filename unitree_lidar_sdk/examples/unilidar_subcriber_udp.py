"""Unitree L1 UDP listener — vectorized point-cloud parsing.

Points are stored as a NumPy structured array. Legacy per-point objects are
still available: indexing or iterating `scan.points` builds a PointUnitree on
demand, so old scripts keep working without the parser paying for them.
"""

import socket
import struct

import numpy as np

UDP_IP = "0.0.0.0"
UDP_PORT = 12345

VERBOSE = True            # printing dominates this loop; turn off in a real pipeline
MAX_DATAGRAM = 10000
RCVBUF_BYTES = 1 << 21    # 2 MiB kernel receive buffer (default is often ~200 KB)

# --------------------------------------------------------------------------
# Wire layout. Struct objects are compiled once instead of re-parsed per call.
# --------------------------------------------------------------------------
HEADER = struct.Struct("=II")           # msgType, length
IMU_MSG = struct.Struct("=dI4f3f3f")    # stamp, id, quat(x,y,z,w), gyro, accel
SCAN_HDR = struct.Struct("=dII")        # stamp, id, validPointsNum

POINT_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("intensity", "<f4"),
    ("time", "<f4"),
    ("ring", "<u4"),
])

POINT_SIZE = POINT_DTYPE.itemsize                   # 24
POINTS_OFFSET = HEADER.size + SCAN_HDR.size         # 24
MAX_POINTS_PER_PACKET = 120

# Cheap guarantee that the NumPy dtype and the C struct agree.
assert POINT_SIZE == struct.calcsize("=fffffI") == 24


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------
class PointUnitree:
    """Unchanged from the original. Built lazily, only when something asks."""

    __slots__ = ("x", "y", "z", "intensity", "time", "ring")

    def __init__(self, x, y, z, intensity, time, ring):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity
        self.time = time
        self.ring = ring

    def __repr__(self):
        return (f"PointUnitree(x={self.x}, y={self.y}, z={self.z}, "
                f"intensity={self.intensity}, time={self.time}, ring={self.ring})")


class PointsView:
    """Sequence facade over a structured point array.

    Old style still works:
        len(scan.points)          -> int
        scan.points[3].x          -> float          (PointUnitree built here)
        for p in scan.points: ... -> PointUnitree

    New style is the fast path:
        scan.points.array         -> np.ndarray[POINT_DTYPE]
        scan.points["x"]          -> float32 column, no copy
        np.asarray(scan.points)   -> the structured array
    """

    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, key):
        if isinstance(key, str):                    # column access: points["x"]
            return self.array[key]
        if isinstance(key, (int, np.integer)):      # legacy object, built on demand
            return PointUnitree(*self.array[key].item())
        return PointsView(self.array[key])          # slice / fancy index

    def __iter__(self):
        for record in self.array:
            yield PointUnitree(*record.item())

    def __array__(self, dtype=None, copy=None):
        out = self.array
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out.copy() if copy else out

    def copy(self):
        """Detach from the reusable receive buffer. Do this before keeping a scan."""
        return PointsView(self.array.copy())

    def tolist(self):
        """Eager list of PointUnitree — exactly the original behaviour, on request."""
        return [PointUnitree(*r.item()) for r in self.array]

    def __repr__(self):
        return f"PointsView({len(self)} points)"


class IMUUnitree:
    __slots__ = ("stamp", "id", "quaternion", "angular_velocity", "linear_acceleration")

    def __init__(self, stamp, id, quaternion, angular_velocity, linear_acceleration):
        self.stamp = stamp
        self.id = id
        self.quaternion = quaternion
        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration


class ScanUnitree:
    __slots__ = ("stamp", "id", "validPointsNum", "points")

    def __init__(self, stamp, id, validPointsNum, points):
        self.stamp = stamp
        self.id = id
        self.validPointsNum = validPointsNum
        self.points = points        # PointsView

    def copy(self):
        return ScanUnitree(self.stamp, self.id, self.validPointsNum, self.points.copy())


# --------------------------------------------------------------------------
# Parsing helpers (no socket involved — unit-testable from raw bytes)
# --------------------------------------------------------------------------
def parse_imu(buf):
    f = IMU_MSG.unpack_from(buf, HEADER.size)
    return IMUUnitree(f[0], f[1], f[2:6], f[6:9], f[9:12])


def parse_scan(buf, nbytes):
    """Return a ScanUnitree, or None if the datagram is short or malformed."""
    if nbytes < POINTS_OFFSET:
        return None
    stamp, scan_id, n = SCAN_HDR.unpack_from(buf, HEADER.size)

    # Never trust a length field from the wire.
    if n > MAX_POINTS_PER_PACKET or POINTS_OFFSET + n * POINT_SIZE > nbytes:
        return None

    # Zero-copy reinterpretation of the payload bytes. This ALIASES `buf` — the
    # next recvfrom_into overwrites it. Call .copy() before keeping the scan.
    array = np.frombuffer(buf, dtype=POINT_DTYPE, count=n, offset=POINTS_OFFSET)
    return ScanUnitree(stamp, scan_id, n, PointsView(array))


def scan_xyz(buf, n):
    """Zero-copy (n, 3) float32 view of just the coordinates.

    The stride is 24 bytes, so this view is not contiguous. KISS-ICP wants a
    contiguous float64 (n, 3), so copy at the handoff:
        np.ascontiguousarray(scan_xyz(buf, n), dtype=np.float64)
    """
    flat = np.frombuffer(buf, dtype="<f4", count=n * 6, offset=POINTS_OFFSET)
    return flat.reshape(n, 6)[:, :3]


# --------------------------------------------------------------------------
# Receive loop
# --------------------------------------------------------------------------
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    sock.bind((UDP_IP, UDP_PORT))

    granted = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"POINT_SIZE = {POINT_SIZE}, IMU size = {IMU_MSG.size}, SO_RCVBUF = {granted}")

    # Reused receive buffer: recvfrom_into writes in place, so the steady-state
    # loop allocates no bytes objects at all.
    packet = bytearray(MAX_DATAGRAM)

    try:
        while True:
            nbytes, addr = sock.recvfrom_into(packet)
            if nbytes < HEADER.size:
                continue

            msgType, length = HEADER.unpack_from(packet, 0)

            if msgType == 101:  # IMU
                if nbytes < HEADER.size + IMU_MSG.size:
                    continue
                imuMsg = parse_imu(packet)

                if VERBOSE:
                    print("An IMU msg is parsed!")
                    print("\tstamp =", imuMsg.stamp, "id =", imuMsg.id)
                    print("\tquaternion (x, y, z, w) =", imuMsg.quaternion)
                    print("\tangular velocity =", imuMsg.angular_velocity)
                    print("\tlinear acceleration =", imuMsg.linear_acceleration)
                    print()

            elif msgType == 102:  # Scan
                scanMsg = parse_scan(packet, nbytes)
                if scanMsg is None:
                    continue

                if VERBOSE:
                    print("A Scan msg is parsed!")
                    print("\tstamp =", scanMsg.stamp, "id =", scanMsg.id)
                    print("\tScan size =", scanMsg.validPointsNum)
                    print("\tfirst 10 points (x, y, z, intensity, time, ring) =")
                    for point in scanMsg.points[:10]:
                        print("\t", point.x, point.y, point.z,
                              point.intensity, point.time, point.ring)
                    print()

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
