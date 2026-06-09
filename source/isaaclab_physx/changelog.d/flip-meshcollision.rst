Changed
^^^^^^^

* **Breaking:** Turned the legacy mesh-collision cfg names into deprecation factories that
  return a fragment list and emit a ``DeprecationWarning``; removal in 5.0. Deprecated
  :class:`MeshCollisionPropertiesCfg` in favor of
  ``[UsdPhysicsMeshCollisionCfg(mesh_approximation_name=...)]``;
  :class:`ConvexHullPropertiesCfg` in favor of
  ``[UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"), PhysxConvexHullCfg(...)]``;
  :class:`ConvexDecompositionPropertiesCfg` in favor of
  ``[UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexDecomposition"), PhysxConvexDecompositionCfg(...)]``;
  :class:`TriangleMeshPropertiesCfg` in favor of
  ``[UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"), PhysxTriangleMeshCfg(...)]``;
  :class:`TriangleMeshSimplificationPropertiesCfg` in favor of
  ``[UsdPhysicsMeshCollisionCfg(mesh_approximation_name="meshSimplification"), PhysxTriangleMeshSimplificationCfg(...)]``;
  and :class:`SDFMeshPropertiesCfg` in favor of
  ``[UsdPhysicsMeshCollisionCfg(mesh_approximation_name="sdf"), PhysxSDFMeshCfg(...)]``.
