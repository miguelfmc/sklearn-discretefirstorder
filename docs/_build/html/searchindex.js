Search.setIndex({"docnames": ["api", "auto_examples/index", "auto_examples/plot_dfo_cv", "auto_examples/plot_dfo_feat_select", "auto_examples/plot_dfo_scaler", "auto_examples/plot_dfo_vs_lasso", "auto_examples/sg_execution_times", "generated/discretefirstorder.DFORegressor", "index", "quick_start", "user_guide"], "filenames": ["api.rst", "auto_examples/index.rst", "auto_examples/plot_dfo_cv.rst", "auto_examples/plot_dfo_feat_select.rst", "auto_examples/plot_dfo_scaler.rst", "auto_examples/plot_dfo_vs_lasso.rst", "auto_examples/sg_execution_times.rst", "generated/discretefirstorder.DFORegressor.rst", "index.rst", "quick_start.rst", "user_guide.rst"], "titles": ["Welcome to <code class=\"docutils literal notranslate\"><span class=\"pre\">sklearn-discretefirstorder</span></code>\u2019s API docs", "General examples", "Cross Validation with DFO Regressor", "DFO Regressor for Subset Selection in a Pipeline", "Scaling input data for the DFO Regressor", "Discrete first-order method vs. Lasso", "Computation times", "<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">discretefirstorder</span></code>.DFORegressor", "Welcome to sklearn-discretefirstorder\u2019s documentation!", "Quick Start with sklearn-discretefirstorder", "User guide : contents"], "terms": {"thi": [0, 2, 3, 5, 8, 9, 10], "i": [0, 2, 5, 8, 9, 10], "an": [3, 4, 7], "exampl": [2, 3, 4, 5], "how": [3, 8], "document": [], "your": 9, "own": [], "project": [8, 9], "introductori": 1, "cross": [1, 6], "valid": [1, 6], "dfo": [1, 6, 7], "regressor": [1, 6, 7], "subset": [1, 2, 6, 8], "select": [1, 2, 6, 8], "pipelin": [1, 6], "discret": [1, 6, 7, 8], "first": [1, 4, 6, 7, 8], "order": [1, 6, 7, 8], "method": [1, 6, 7, 8], "v": [1, 2, 6], "lasso": [1, 6], "download": [1, 2, 3, 4, 5], "all": [1, 3], "python": [1, 2, 3, 4, 5], "sourc": [1, 2, 3, 4, 5, 7], "code": [1, 2, 3, 4, 5], "auto_examples_python": 1, "zip": 1, "jupyt": [1, 2, 3, 4, 5], "notebook": [1, 2, 3, 4, 5], "auto_examples_jupyt": 1, "galleri": [1, 2, 3, 4, 5], "sphinx": [1, 2, 3, 4, 5], "click": [2, 3, 4, 5], "here": [2, 3, 4, 5], "full": [2, 3, 4, 5], "us": [2, 4], "cv": [], "find": [], "right": 2, "number": [2, 7], "featur": 7, "keep": [2, 3, 5, 7], "todo": [], "creat": [], "total": [2, 3, 4, 5, 6], "run": [2, 3, 4, 5, 7], "time": [2, 3, 4, 5], "script": [2, 3, 4, 5], "0": [2, 3, 4, 5, 6, 7], "minut": [2, 3, 4, 5], "064": [], "second": [2, 3, 4, 5], "plot_dfo_cv": [2, 6], "py": [2, 3, 4, 5, 6], "ipynb": [2, 3, 4, 5], "gener": [2, 3, 4, 5], "includ": 3, "dforegressor": [2, 3, 5, 10], "part": [], "sklearn": [2, 3, 4, 5], "000": 6, "plot_dfo_feat_select": [3, 6], "support": 5, "recoveri": 5, "regress": [5, 7], "import": [2, 3, 4, 5, 7, 10], "numpi": [2, 4, 5, 7, 10], "np": [2, 4, 5, 7, 10], "matplotlib": [2, 4, 5], "pyplot": [2, 4, 5], "plt": [2, 4, 5], "from": [2, 3, 4, 5, 7, 10], "make_regress": [2, 3, 4, 5], "linear_model": 5, "model_select": [2, 3, 4, 5], "train_test_split": [2, 3, 4, 5], "discretefirstord": [2, 3, 4, 5, 10], "we": [2, 3, 4, 5], "onli": [2, 3, 4, 5], "10": [2, 3, 4, 5], "inform": [2, 3, 4, 5, 8], "true": [2, 3, 4, 5, 7], "x": [2, 3, 4, 5, 7, 10], "y": [2, 3, 4, 5, 7, 10], "coef": [2, 3, 4, 5], "n_sampl": [2, 3, 5, 7], "10000": [2, 3, 4, 5], "n_featur": [2, 3, 5, 7], "30": [2, 3, 4, 5], "n_inform": [2, 3, 5], "bia": [2, 3, 5], "5": [2, 3, 4, 5], "nois": [2, 3, 5], "random_st": [2, 3, 4, 5, 7], "11": [2, 3, 5], "x_train": [2, 3, 4, 5], "x_test": [2, 3, 4, 5], "y_train": [2, 3, 4, 5], "y_test": [2, 3, 4, 5], "test_siz": [2, 3, 4, 5], "15": [2, 3, 4, 5], "42": [2, 3, 4, 5], "fit": [2, 5, 7], "lasso_scor": 5, "score": [3, 4, 5], "lasso_coef_error": 5, "sqrt": [4, 5], "sum": [4, 5], "coef_": [4, 5, 7], "2": [2, 4, 5], "k": [3, 4, 5, 7], "fit_intercept": [4, 5, 7], "normal": [4, 5, 7, 10], "dfo_scor": [4, 5], "dfo_coef_error": [4, 5], "can": [2, 4, 5, 9, 10], "see": 5, "case": [2, 3, 5], "seem": [2, 5], "recov": 5, "more": 5, "faithfulli": 5, "ground": 5, "truth": 5, "fig": [2, 5], "ax": [2, 5], "subplot": [2, 5], "1": [2, 3, 4, 5, 7, 10], "figsiz": [2, 5], "9": [2, 4, 5], "12": 5, "index": 5, "arang": [5, 7, 10], "shape": [5, 7], "bar_width": 5, "25": 5, "barh": 5, "label": 5, "f": [3, 4, 5], "r2": 5, "4f": [3, 4, 5], "error": [4, 5], "2f": 5, "set_xlabel": [2, 5], "valu": [5, 7], "set_ylabel": [2, 5], "set_titl": [2, 5], "compar": [4, 5], "set_ytick": 5, "set_yticklabel": 5, "feat_": 5, "str": [5, 7], "rang": 5, "grid": 5, "legend": 5, "show": [2, 5], "700": [], "plot_dfo_vs_lasso": [5, 6], "00": 6, "764": [], "execut": 6, "auto_exampl": 6, "file": 6, "mb": 6, "scikit": [3, 8], "learn": [3, 7, 8], "compat": 8, "implement": [0, 7, 8], "simpl": 8, "base": [3, 8], "work": 8, "bertsima": 8, "et": 8, "al": 8, "regard": 8, "refer": 8, "A": [2, 8], "set": [3, 4, 7, 8], "you": [9, 10], "easili": 10, "pip": 9, "If": [4, 9], "prefer": 9, "clone": 9, "machin": 9, "local": 9, "copi": 9, "repo": 9, "git": 9, "http": 9, "github": 9, "com": 9, "miguelfmc": 9, "cd": 9, "feel": 9, "free": 9, "rais": 9, "pull": 9, "request": 9, "kei": 10, "packag": [0, 10], "follow": 10, "100": [7, 10], "reshap": [7, 10], "random": [3, 4, 7, 10], "size": [2, 4, 7, 10], "scale": [1, 6, 7], "input": [1, 6, 7], "data": [1, 2, 6, 7], "ensembl": 3, "feature_select": 3, "selectfrommodel": 3, "coeffici": [2, 3, 4, 7], "comparison": [2, 3], "estim": [2, 3, 7], "out": 3, "box": 3, "rf": 3, "origin": [3, 4], "our": [2, 3], "train": [3, 7], "let": 3, "": 3, "long": 3, "take": [3, 4, 7], "r\u00b2": [2, 3, 4], "test": [3, 4], "500": 3, "t0": 3, "perf_count": 3, "delta": 3, "rf_time": 3, "rf_score": 3, "print": [2, 3, 4], "forest": 3, "68": [], "9782": [], "9124": 3, "now": [3, 4], "construct": 3, "step": [3, 7], "veri": 3, "small": 3, "threshold": 3, "non": [2, 3, 7], "zero": [2, 3, 4, 7], "selector": 3, "1e": 3, "pipeline_tim": 3, "pipeline_scor": 3, "23": [], "9407": [], "9329": 3, "33": [], "301": [], "intern": 4, "conjunct": 4, "scaler": 4, "e": 4, "g": 4, "standardscal": [], "003": [], "plot_dfo_scal": [4, 6], "01": [], "304": [], "69": [], "4522": [], "24": 3, "1680": [], "34": [], "014": [], "001": 7, "016": [], "gridsearchcv": 2, "search": [], "fold": [], "optim": 7, "cardin": [], "paramet": 7, "priori": 2, "don": [2, 4], "t": [2, 4], "know": 2, "sweep": 2, "through": 2, "list": 2, "possibl": 2, "perform": 2, "provid": 2, "best": 2, "As": 2, "expect": 2, "achiev": 2, "4": [2, 4], "6": [2, 4], "8": [2, 4], "20": [2, 4], "clf": 2, "best_estimator_": 2, "best_score_": 2, "9988501325091541": 2, "visual": [], "mean": 4, "each": 7, "hyperparamet": [], "param_valu": 2, "cv_results_": 2, "param_k": 2, "astyp": 2, "int64": 2, "mean_test_scor": 2, "plot": 2, "o": 2, "473": [], "05": [], "924": [], "434": [], "06": [], "358": [], "64": [], "6148": [], "2661": [], "28": [], "7": [2, 4], "211": 2, "62": 3, "4051": 3, "21": 3, "5355": 3, "289": 3, "preprocess": 4, "randint": 4, "std": 4, "cov": 4, "diag": 4, "epsilon": 4, "seed": 4, "multivariate_norm": 4, "concaten": 4, "arrai": [4, 7], "3": 4, "care": 4, "9920": [], "model": 4, "fals": 4, "pipeline_intercept": 4, "need": 4, "rescal": 4, "rescaled_coef": 4, "axi": 4, "pipeline_intercept_scor": 4, "pipeline_intercept_coef_error": 4, "extern": 4, "1225": [], "want": 4, "term": [4, 7], "same": 4, "center": 4, "target": [4, 7], "pipeline_no_intercept": 4, "pipeline_no_intercept_scor": 4, "pipeline_no_intercept_coef_error": 4, "9511": [], "479": [], "372": 5, "32": [], "351": [], "07": [], "9913": 4, "0531": 4, "3419": 4, "828": [4, 6], "guid": 0, "At": 9, "moment": 9, "avail": 9, "class": 7, "loss": 7, "mse": 7, "learning_r": 7, "auto": 7, "polish": 7, "n_run": 7, "50": 7, "max_it": 7, "tol": 7, "none": 7, "type": 7, "minim": 7, "One": 7, "mae": 7, "float": 7, "rate": 7, "int": 7, "bool": 7, "whether": 7, "least": 7, "squar": 7, "activ": 7, "procedur": 7, "maximum": 7, "dure": 7, "one": 7, "algorithm": 7, "toler": 7, "below": 7, "which": 7, "stop": 7, "intercept": 7, "attribut": 7, "ndarrai": 7, "vector": 7, "intercept_": 7, "__init__": 7, "coef_init": 7, "like": 7, "sampl": 7, "option": 7, "initi": 7, "return": 7, "self": 7, "object": 7, "predict": 7, "The": 7, "output": 7, "correspond": 7}, "objects": {"discretefirstorder": [[7, 0, 1, "", "DFORegressor"]], "discretefirstorder.DFORegressor": [[7, 1, 1, "", "__init__"], [7, 1, 1, "", "fit"], [7, 1, 1, "", "predict"]]}, "objtypes": {"0": "py:class", "1": "py:method"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"]}, "titleterms": {"sklearn": [0, 8, 9], "discretefirstord": [0, 7, 8, 9], "api": [0, 8], "gener": 1, "exampl": [1, 7, 8], "cross": 2, "valid": 2, "dfo": [2, 3, 4, 5, 10], "regressor": [2, 3, 4, 10], "subset": [3, 10], "select": [3, 10], "pipelin": [3, 4], "discret": [0, 3, 5, 10], "first": [0, 3, 5, 10], "order": [0, 3, 5, 10], "method": [3, 5, 10], "v": 5, "lasso": 5, "creat": [2, 3, 4, 5], "synthet": [2, 3, 4, 5], "dataset": [2, 3, 4, 5], "some": [2, 3, 4, 5], "uninform": [2, 3, 4, 5], "featur": [2, 3, 4, 5], "comparison": 5, "estim": [0, 5, 10], "coeffici": 5, "comput": 6, "time": 6, "welcom": [0, 8], "": [0, 8], "document": 8, "get": 8, "start": [8, 9], "user": [8, 10], "guid": [8, 10], "quick": 9, "instal": 9, "packag": 9, "from": 9, "pypi": 9, "sourc": 9, "contribut": 9, "The": 10, "fit": [3, 4, 10], "randomforestregressor": 3, "directli": [3, 4], "data": [3, 4], "scale": 4, "input": 4, "grid": 2, "search": 2, "k": 2, "fold": 2, "find": 2, "optim": 2, "cardin": 2, "paramet": 2, "visual": 2, "mean": 2, "cv": 2, "score": 2, "each": 2, "valu": 2, "hyperparamet": 2, "us": [3, 7], "part": 3, "dforegressor": [4, 7], "standardscal": 4, "intercept": 4, "doc": 0}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "sphinx": 56}})