/** Structure-of-arrays coordinates used only for correspondence. */
export type Coordinates = [Float32Array, Float32Array, Float32Array];

/** Coordinate tie-breaks recover the same partition after an input permutation. */
function compareIndex(a: number, b: number, coordinates: Coordinates, axis: number): number {
  return coordinates[axis][a] - coordinates[axis][b]
    || coordinates[(axis + 1) % 3][a] - coordinates[(axis + 1) % 3][b]
    || coordinates[(axis + 2) % 3][a] - coordinates[(axis + 2) % 3][b]
    || a - b;
}

/** Partition indices at a quantile; each recursion level does expected O(N) work. */
function selectIndex(order: Uint32Array, coordinates: Coordinates, axis: number, start: number, end: number, rank: number): void {
  let left = start;
  let right = end - 1;
  const values = coordinates[axis];
  while (left < right) {
    const middle = (left + right) >>> 1;
    const x = order[left], y = order[middle], z = order[right];
    const pivot = compareIndex(x, y, coordinates, axis) < 0
      ? (compareIndex(y, z, coordinates, axis) < 0 ? y : (compareIndex(x, z, coordinates, axis) < 0 ? z : x))
      : (compareIndex(x, z, coordinates, axis) < 0 ? x : (compareIndex(y, z, coordinates, axis) < 0 ? z : y));
    const pivotValue = values[pivot];
    let lo = left;
    let hi = right;
    while (lo <= hi) {
      while (values[order[lo]] < pivotValue || (values[order[lo]] === pivotValue && compareIndex(order[lo], pivot, coordinates, axis) < 0)) lo++;
      while (values[order[hi]] > pivotValue || (values[order[hi]] === pivotValue && compareIndex(order[hi], pivot, coordinates, axis) > 0)) hi--;
      if (lo <= hi) {
        const index = order[lo]; order[lo++] = order[hi]; order[hi--] = index;
      }
    }
    if (rank <= hi) right = hi;
    else if (rank >= lo) left = lo;
    else return;
  }
}

/** Shared deterministic lockstep bisection for sample scoring and full pairing. */
export function matchCoordinates(
  coordinatesA: Coordinates,
  coordinatesB: Coordinates,
  emit: (indexA: number, indexB: number, visibleA: boolean, visibleB: boolean) => void,
): void {
  const countA = coordinatesA[0].length;
  const countB = coordinatesB[0].length;
  const orderA = new Uint32Array(countA);
  const orderB = new Uint32Array(countB);
  for (let i = 0; i < countA; i++) orderA[i] = i;
  for (let i = 0; i < countB; i++) orderB[i] = i;
  const [ax, ay, az] = coordinatesA;
  const [bx, by, bz] = coordinatesB;
  function visit(startA: number, endA: number, startB: number, endB: number): void {
    const lengthA = endA - startA;
    const lengthB = endB - startB;
    if (lengthA === 1 || lengthB === 1) {
      if (lengthA === 1 && lengthB === 1) {
        emit(orderA[startA], orderB[startB], true, true);
        return;
      }
      const singletonA = lengthA === 1;
      const singleton = singletonA ? orderA[startA] : orderB[startB];
      const one = singletonA ? coordinatesA : coordinatesB;
      const many = singletonA ? coordinatesB : coordinatesA;
      const order = singletonA ? orderB : orderA;
      const start = singletonA ? startB : startA;
      const end = singletonA ? endB : endA;
      let nearest = start;
      let nearestDistance = Infinity;
      for (let i = start; i < end; i++) {
        const index = order[i];
        const dx = one[0][singleton] - many[0][index];
        const dy = one[1][singleton] - many[1][index];
        const dz = one[2][singleton] - many[2][index];
        const distance = dx * dx + dy * dy + dz * dz;
        if (distance < nearestDistance) { nearest = i; nearestDistance = distance; }
      }
      for (let i = start; i < end; i++) {
        if (singletonA) emit(singleton, order[i], i === nearest, true);
        else emit(order[i], singleton, true, i === nearest);
      }
      return;
    }

    // Read each index once for all three union dimensions (hot worker loop).
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = startA; i < endA; i++) {
      const index = orderA[i];
      const x = ax[index], y = ay[index], z = az[index];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    for (let i = startB; i < endB; i++) {
      const index = orderB[i];
      const x = bx[index], y = by[index], z = bz[index];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    let axis = maxY - minY > maxX - minX ? 1 : 0;
    if (maxZ - minZ > (axis === 0 ? maxX - minX : maxY - minY)) axis = 2;
    if (lengthA === 2 && lengthB === 2) {
      const a0 = orderA[startA], a1 = orderA[startA + 1];
      const b0 = orderB[startB], b1 = orderB[startB + 1];
      const forwardA = compareIndex(a0, a1, coordinatesA, axis) < 0;
      const forwardB = compareIndex(b0, b1, coordinatesB, axis) < 0;
      emit(forwardA ? a0 : a1, forwardB ? b0 : b1, true, true);
      emit(forwardA ? a1 : a0, forwardB ? b1 : b0, true, true);
      return;
    }
    // Keep both sides nonempty until a singleton can provide local padding.
    const larger = Math.max(lengthA, lengthB);
    const fraction = Math.floor(larger / 2) / larger;
    const splitA = startA + Math.max(1, Math.min(lengthA - 1, Math.round(lengthA * fraction)));
    const splitB = startB + Math.max(1, Math.min(lengthB - 1, Math.round(lengthB * fraction)));
    selectIndex(orderA, coordinatesA, axis, startA, endA, splitA);
    selectIndex(orderB, coordinatesB, axis, startB, endB, splitB);
    visit(startA, splitA, startB, splitB);
    visit(splitA, endA, splitB, endB);
  }

  visit(0, countA, 0, countB);
}
