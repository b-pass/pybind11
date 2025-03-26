import random
import subprocess
import itertools
from collections import defaultdict
import importlib.machinery
import os
import sys
import time
import cython
from matplotlib.patches import Rectangle
    
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 5.0 
cycle = [x['color'] for x in mpl.rcParams['axes.prop_cycle']]

experiment_labels = []
sizes = defaultdict(lambda: [])
times = defaultdict(lambda: [])

from sysconfig import get_paths as gp
suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

nanobind_path = '/workspace/nanobind/include'
pybind11_path = '/workspace/pybind11/include'
pybind11_branch_path = '/workspace/pybind11_branch/include'

# Path to boost (in this case, assumed to be installed by the OS)
boost_path = '/usr/include/boost'

cmd_base = ['clang++', '-march=native', '-shared', '-rpath', '..', '-std=c++17', '-I', '../include', '-I', gp()['include'],
            '-Wno-deprecated-declarations', '-fPIC', f'-L{boost_path}/stage/lib', '-L..', '-fno-stack-protector',
            '-DPYBIND11_USE_SMART_HOLDER_AS_DEFAULT']

def gen_file(name, func, libs=('cython', 'boost', 'pybind11', 'pybind11_branch', 'nanobind')):
    for i, lib in enumerate(libs):    
        for opt_mode, opt_flags in {'debug' : ['-O0', '-g3'], 'opt' : ['-O3', '-g0']}.items():
            if lib != 'cython':
                fname = name + '_' + lib + '.cpp'
            else:
                fname = name + '_' + lib + '_' + opt_mode + '.pyx'

            with open(fname, 'w') as f:
                if lib == 'boost':
                    f.write(f'#include <boost/python.hpp>\n')
                    f.write(f'namespace py = boost::python;\n\n')
                    f.write(f'BOOST_PYTHON_MODULE({name}_{lib}_{opt_mode}) {{\n')
                elif lib == 'nanobind':
                    f.write(f'#include <nanobind/nanobind.h>\n\n')
                    f.write(f'namespace py = nanobind;\n\n')
                    f.write(f'NB_MODULE({name}_{lib}_{opt_mode}, m) {{\n')
                elif lib.startswith('pybind11'):
                    f.write(f'#include <pybind11/pybind11.h>\n\n')
                    f.write(f'namespace py = pybind11;\n\n')
                    f.write(f'PYBIND11_MODULE({name}_{lib}_{opt_mode}, m) {{\n')
                elif lib == 'cython':
                    f.write(f'from libc.stdint cimport uint16_t, int32_t, uint32_t, int64_t, uint64_t\n')

                func(f, lib)
                if lib != 'cython':
                    f.write(f'}}\n')

            fname_out = name + '_' + lib + '_' + opt_mode  + suffix
            cmd = cmd_base + opt_flags + [name + '_' + lib + '.cpp', '-o', fname_out]
            if lib == 'nanobind':
                cmd += ['-I', nanobind_path, '-L',nanobind_path+"/../build/tests","-lnanobind-static"]
            elif lib == 'boost':
                cmd += ['-I', boost_path, '-lboost_python312']
            elif lib == 'pybind11':
                cmd += ['-I', pybind11_path]
            elif lib == 'pybind11_branch':
                cmd += ['-I', pybind11_branch_path]
             
            print(' '.join(cmd))
            time_list = []
            for l in range(5):
                if 'skip-compile' in sys.argv and os.path.exists(fname_out):
                   time_list.append(0.00001)
                   continue
                time_before = time.perf_counter()
                if lib == 'cython':
                    subprocess.check_call(['cython3', '-3',  '--cplus', fname, '-o', name + '_' + lib + '.cpp'])
                subprocess.check_call(cmd)
                time_after = time.perf_counter()
                time_list.append(time_after-time_before)
            time_list.sort()
            
            if opt_mode != 'debug':
                subprocess.check_call(['strip', fname_out])
            if i == 0:
                experiment_labels.append(name + ' [' + opt_mode + ']')
            sizes[lib].append(os.path.getsize(fname_out) / (1024 * 1024))
            times[lib].append(time_list[len(time_list)//2])


            
def gen_func(f, lib):
    types = [ 'uint16_t', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t', 'float' ]
    if lib == 'boost':
        prefix = 'py::'
    else:
        prefix = 'm.'
    for i, t in enumerate(itertools.permutations(types)):
        args = f'{t[0]} a, {t[1]} b, {t[2]} c, {t[3]} d, {t[4]} e, {t[5]} f'
        if lib != 'cython':
            f.write('    %sdef("test_%04i", +[](%s) { return a+b+c+d+e+f; });\n' % (prefix, i, args))
        else:
            f.write('cpdef float test_%04i(%s):\n    return a+b+c+d+e+f\n\n' % (i, args))


def gen_class(f, lib):
    types = [ 'uint16_t', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t', 'float' ]

    for i, t in enumerate(itertools.permutations(types)):
        if lib == 'boost':
            prefix = ''
            postfix = f', py::init<{t[0]}, {t[1]}, {t[2]}, {t[3]}, {t[4]}, {t[4]}>()'
            func_prefix = 'py::def'

        else:
            prefix = 'm, '
            postfix = ''
            func_prefix = 'm.def'

        if lib != 'cython':
            f.write(f'    struct Struct{i} {{\n')
            f.write(f'        {t[0]} a; {t[1]} b; {t[2]} c; {t[3]} d; {t[4]} e; {t[5]} f;\n')
            f.write(f'        Struct{i}({t[0]} a, {t[1]} b, {t[2]} c, {t[3]} d, {t[4]} e, {t[5]} f) : a(a), b(b), c(c), d(d), e(e), f(f) {{ }}\n')
            f.write(f'        float sum() const {{ return a+b+c+d+e+f; }}\n')
            f.write(f'    }};\n')
        else:
            f.write(f'cdef class Struct{i}:\n')
            f.write(f'    cdef {t[0]} a\n')
            f.write(f'    cdef {t[1]} b\n')
            f.write(f'    cdef {t[2]} c\n')
            f.write(f'    cdef {t[3]} d\n')
            f.write(f'    cdef {t[4]} e\n')
            f.write(f'    cdef {t[5]} f\n\n')
            f.write(f'    def __cinit__(self, {t[0]} a, {t[1]} b, {t[2]} c, {t[3]} d, {t[4]} e, {t[5]} f):\n')
            f.write(f'        self.a = a\n')
            f.write(f'        self.b = b\n')
            f.write(f'        self.c = c\n')
            f.write(f'        self.d = d\n')
            f.write(f'        self.e = e\n')
            f.write(f'        self.f = f\n\n')
            f.write(f'    cpdef float sum(self):\n')
            f.write(f'        return self.a+self.b+self.c+self.d+self.e+self.f\n\n')
            continue

        f.write(f'    py::class_<Struct{i}>({prefix}\"Struct{i}\"{postfix})\n')
        
        if lib != 'boost':
                f.write(f'        .def(py::init<{t[0]}, {t[1]}, {t[2]}, {t[3]}, {t[4]}, {t[5]}>())\n')
        f.write(f'        .def("sum", &Struct{i}::sum);\n\n')
        
        if i > 250:
            break;
        
        
gen_file('func', gen_func)
gen_file('class', gen_class)
experiment_labels = ['func [debug]', 'func [opt]', 'class [debug]', 'class [opt]']

print(experiment_labels)
print(dict(sizes))
print(dict(times))

plot_colors = {
    'boost': cycle[1],
    'pybind11': cycle[3],
    'pybind11_branch': cycle[5],
    'cython' : cycle[4],
    'nanobind': cycle[0]
}
plot_labels = {
    'boost' : 'Boost.Python',
    'pybind11' : 'pybind11',
    'pybind11_branch' : 'pybind11_branch',
    'cython' : 'Cython',
    'nanobind' : 'nanobind'
}

def bars(data, ylim_scale = 1, figsize_scale = 1, width_scale=1.0, debug_shift=0.1):
    ylim = 0
    for n, d in data.items():
        if len(d) == 0:
            continue
        ylim = max(max(d), ylim)
    ylim *= ylim_scale * 1.3

    def adj(ann):
        for i, a in enumerate(ann):
            if a.xy[1] > ylim*.9:
                a.xy = (a.xy[0], ylim * 0.8)
                if i%2 == 1:
                    a.set_color('white')

    fig, ax = plt.subplots(figsize=[11.25*figsize_scale, 3*figsize_scale])
    width = 1.0/(len(data) + 1)*width_scale
    x = np.arange(4)

    result = []
    for i, n in enumerate(plot_labels):
        d = data[n]
        if len(d) == 0:
            continue

        col = plot_colors[n]
        if col != 'None':
            kwargs = { 'edgecolor': 'black', 'color': col }
        else:
            kwargs =  {'edgecolor': 'white', 'hatch' : '/', 'color':cycle[7]}
            
        bar = ax.bar(x+width*(i -(len(data)-1)/2), d, width, label=plot_labels[n], align='center', **kwargs)
        result.append(bar)
        
    ax.add_patch(Rectangle((-0.65+debug_shift, -1), 1, 28, facecolor='white', alpha=.8, edgecolor='None'))
    ax.add_patch(Rectangle((1.4+debug_shift, -1), 1, 25, facecolor='white', alpha=.8, edgecolor='None'))

    for i, n in enumerate(plot_labels):
        d = data[n]
        if len(d) == 0:
            continue

        bar = result[i]
        if n == 'nanobind':
            adj(ax.bar_label(bar, fmt='%.2f'))
        else:
            improvement = np.array(d) / np.array(data['nanobind'])
            improvement = ['%.2f\n(x%.1f)' % (d[i], v) for i, v in enumerate(improvement)]
            adj(ax.bar_label(bar, labels=improvement, padding=3))
        
    ax.set_ylim(0, ylim)
    ax.set_xticks(x, experiment_labels)
    return fig, ax

if 'skip-compile' not in sys.argv:
    fig, ax = bars(times, ylim_scale=0.93, figsize_scale=1.1, width_scale=1)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Compilation time')
    ax.set_xlim(-0.45,3.45)

    ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig('times.png', facecolor='white', dpi=200, bbox_inches='tight', pad_inches = 0)
    plt.savefig('times.svg', facecolor='white', bbox_inches='tight', pad_inches = 0)
    plt.show()

fig, ax = bars(sizes, ylim_scale=.085, figsize_scale=1.1)
ax.set_ylabel('Size (MiB)')
ax.set_title('Binary size')
ax.set_xlim(-0.45,3.45)
ax.legend(loc='lower left')

fig.tight_layout()
plt.savefig('sizes.png', facecolor='white', dpi=200, bbox_inches='tight', pad_inches = 0)
plt.savefig('sizes.svg', facecolor='white', bbox_inches='tight', pad_inches = 0)
plt.show()

try:
    import cppyy
    if not hasattr(cppyy.gbl, 'test_0000'):
        cppyy.include('cppyy.h')
except ImportError:
    cppyy = None

plot_colors = {
    'boost': cycle[1],
    'cython': cycle[4],
    'pybind11': cycle[3],
    'cppyy' : cycle[8],
    'python': 'None',
    'nanobind': cycle[0]
}

plot_labels = {
    'boost' : 'Boost.Python',
    'pybind11' : 'pybind11',    
    'cython' : 'Cython',
    'nanobind' : 'nanobind',
    'python' : 'Python'
}

if cppyy is not None:
    plot_labels['cppyy'] = 'cppyy'

class native_module:
    @staticmethod
    def test_0000(a, b, c, d, e, f):
        return a + b + c + d +e + f

    
    class Struct0:
        def __init__(self, a, b, c, d, e, f):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.e = e
            self.f = f

        def sum(self):
            return self.a + self.b + self.c + self.e + self.f
    

rtimes = defaultdict(lambda: [])
for name in ['func', 'class']:
    its = 10000000 if name == 'func' else 2500000
    for lib in plot_labels:
        for mode in ['debug', 'opt']:
            print(mode,lib,name)
            if lib == 'cppyy':
                m = cppyy.gbl
            elif lib == 'python':
                m = native_module
            else:
                m = importlib.import_module(f'{name}_{lib}_{mode}')
         
            time_list = []
            for i in range(5):
                time_before = time.perf_counter()
                if name == 'func':
                    func = m.test_0000
                    for i in range(its):
                        func(1,2,3,4,5,6)
                elif name == 'class':
                    cls = m.Struct0
                    sum_member = cls.sum
                    for i in range(its):
                        sum_member(cls(1,2,3,4,5,6))

                time_after = time.perf_counter()
                time_list.append(time_after-time_before)
            time_list.sort()

            rtimes[lib].append(time_list[len(time_list)//2])

fig, ax = bars(rtimes, ylim_scale=.188, figsize_scale=1.25, width_scale=1, debug_shift=.1)
ax.set_ylabel('Time (seconds)')
ax.set_title('Runtime performance')
ax.set_xlim(-0.45,3.45)
ax.legend()
fig.tight_layout()
plt.savefig('perf.png', facecolor='white', dpi=200, bbox_inches='tight', pad_inches = 0)
plt.savefig('perf.svg', facecolor='white', bbox_inches='tight', pad_inches = 0)
plt.show()


