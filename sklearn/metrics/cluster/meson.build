py.extension_module(
  '_expected_mutual_info_fast',
  cython_gen.process('_expected_mutual_info_fast.pyx'),
  subdir: 'sklearn/metrics/cluster',
  install: true
)

py.extension_module(
  '_dbcv_linkage',
  '_dbcv_linkage.pyx',
  dependencies: [np_dep],
  cython_args: cython_args,
  subdir: 'sklearn/metrics/cluster',
  install: true
)
