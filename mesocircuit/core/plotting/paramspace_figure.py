import os
import matplotlib.pyplot as plt
import subprocess
import pickle
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')

def parameterspace_overviews(paramspace_key, data_dir, hspace=0.7, wspace=0.5):
    """
    TODO use auto data dir when mesocircuit_framework is revised

    Creates overview figures to compare single parameter set figures across
    parameter space.

    Uses 'pdfinfo' and 'pdflatex'.

    Parameters
    ----------
    paramspace_key
        Key of parameter space.
    data_dir
        Absolute path to data directory for reading from and writing to.
    hspace
        Height space between panels (in inch), must provide sufficient space for
        parameterset titles.
    wspace
        Width space between panels (in inch).
    """
    print(f'Plotting overviews for parameter space {paramspace_key}.')

    def __strfmt(raw):
        """ Converts to string and replaces underscore. """
        if isinstance(raw, float):
            fmt = f'{raw:.3f}'
        if isinstance(raw, int):
            fmt = str(raw)
        elif isinstance(raw, str):
            fmt = '\_'.join(raw.split('_'))
        return fmt

    # load parameter space view, ranges and hash map
    with open(os.path.join(data_dir, paramspace_key, 'parameter_space',
                           'parameters', 'psview_dict.pkl'), 'rb') as f:
        psview = pickle.load(f)
    with open(os.path.join(data_dir, paramspace_key, 'parameter_space',
                           'parameters', 'ranges_hashmap.pkl'), 'rb') as f:
        hashmap_ranges = pickle.load(f)
    ranges = hashmap_ranges['ranges']
    hashmap = hashmap_ranges['hashmap']

    # infer 2D indices for figure
    shape = np.shape(hashmap)
    rows = shape[0]
    cols = shape[1]
    indices = np.zeros(shape, dtype=object)
    for r in np.arange(rows):
        for c in np.arange(cols):
            indices[r,c] = (r,c)
    indices = indices.flatten()
    print(indices)

    # existing single figures (exclude parameter_space folder)
    sfigs = glob.glob(os.path.join(data_dir, paramspace_key,
                                   '[!parameter_space]*', 'plots', '*'))
    sfigs = np.unique([os.path.basename(p) for p in sfigs])
    if len(sfigs) == 0:
        print('  No single figures found.')

    for sf in sfigs:
        print(f'  Plotting {sf} for parameter space.')
        name, extension = sf.split('.')
        iterate_sf = True
        for ind in indices:
            try:
                sfig_name = os.path.join(data_dir, paramspace_key,
                                         hashmap[ind], 'plots', sf)
                break
            except:
                print(f'Single figures {sf} do not exist in parameter space.')
                iterate_sf = False
        # stop processing of non-existing single figure
        if iterate_sf == False:
            break

        # figure size
        if extension == 'pdf':
            pdfinfo = subprocess.check_output(['pdfinfo', sfig_name]).decode('utf-8')
            for line in pdfinfo.split('\n'):
                if 'Page size' in line:
                    ps = line
            ps = ps.split(':')[1].split('pts')[0].split('x')
            sfig_size_pts = [float(s) for s in ps]
            sfig_size = [pts / 72 for pts in sfig_size_pts] # to inch
        else:
            raise Exception (
                f'Size of {extension} figures cannot be inferred.')

        fig_size = (sfig_size[0] * cols + wspace * (cols - 1),
                    sfig_size[1] * rows + hspace * rows)

        # write tex script
        fname = os.path.join(data_dir, paramspace_key, 'parameter_space',
                             'plots', sf.split('.')[0])
        file = open('%s.tex' % fname , 'w')
        file.write(r"\documentclass{article}")
        file.write("\n")
        file.write(r"\usepackage{geometry}")
        file.write("\n")
        file.write(r"\geometry{paperwidth=%.3fin, paperheight=%.3fin, top=0pt, bottom=0pt, right=0pt, left=0pt}" % (fig_size[0],fig_size[1]))
        file.write("\n")
        file.write(r"\usepackage{tikz}")
        file.write("\n")
        file.write(r"\pagestyle{empty}")
        file.write("\n")
        file.write(r"\begin{document}")
        file.write("\n")
        file.write(r"\noindent")
        file.write("\n")
        file.write(r"\resizebox{\paperwidth}{!}{")
        file.write("\n")
        file.write(r"  \begin{tikzpicture}")
        file.write("\n")

        # iterate over parametersets and insert single figures
        for ind in indices:
            # label
            h = hashmap[ind]
            label = '[align=left]' + h
            for i,r in enumerate(ranges):
                label += r"\\ "
                val = psview[paramspace_key]['paramsets'][h][r[0]][r[1]]
                label += f'{__strfmt(r[0])}[{__strfmt(r[1])}] = {__strfmt(val)}'

            # position
            pos = (-0.5*fig_size[0] + sfig_size[0] * (0.5 + ind[1]) + wspace * ind[1],
                   0.5*fig_size[1] - sfig_size[1] * (0.5 + ind[0]) - hspace * (ind[0] + 1))

            file.write(r"    \node[label={%s},inner sep=-1pt,rectangle] at (%.4fin,%.4fin)" % (label, pos[0],pos[1]))

            # single figure file name
            sfig_name = os.path.join(data_dir, paramspace_key,
                                         hashmap[ind], 'plots', sf)
            file.write("\n")
            file.write(r"    {\includegraphics{%s}};" % (sfig_name))
            file.write("\n")

        # draw grid
        for x in np.arange(cols - 1):
            xval = -0.5 * (fig_size[0]+wspace) + (sfig_size[0]+ wspace) * (x+1)
            file.write(r"    \draw ({%.4fin},{%.4fin}) --({%.4fin},{%.4fin});" % (
                xval, -0.5 * fig_size[1], xval, 0.5 * fig_size[1]))
            file.write("\n")
        for y in np.arange(rows - 1):
            yval = 0.5 * fig_size[1] - (hspace + sfig_size[1]) * (y + 1)
            file.write(r"    \draw ({%.4fin},{%.4fin}) --({%.4fin},{%.4fin});" % (
                -0.5 * fig_size[0], yval, 0.5 * fig_size[0], yval))
            file.write("\n")

        file.write(r"  \end{tikzpicture}")
        file.write("\n")
        file.write(r"}")
        file.write("\n")
        file.write(r"\end{document}")
        file.write("\n")

        file.close()

        # execute tex script
        os.system('pdflatex -output-directory=%s %s.tex' % (
            os.path.join(data_dir, paramspace_key, 'parameter_space','plots'),
            fname))

        # remove unnecessary files
        for ext in ['aux', 'log', 'tex']:
            os.system('rm ' + 
                os.path.join(data_dir, paramspace_key, 'parameter_space','plots',
                f'{fname}.{ext}'))
    return
