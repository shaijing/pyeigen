project(
  'pyeigen',
  ['c', 'cpp'],
  version: '1.0.0',
  default_options: [
    'buildtype=release',
    'c_std=c11',
    'cpp_std=c++20',
  ],
)

# py = import('python').find_installation('python3.12',pure: false)
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

subdir('pyeigen')