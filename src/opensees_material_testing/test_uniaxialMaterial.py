"""
Test a uniaxialMaterial in OpenSeesPy using modulated white-noise strain history.

Workflow: fix all nodes in the existing model, build a side 1D model with the
material and run the test, then remove the side model and restore the original
model (NDM/NDF and node fixities).

Reference: https://portwooddigital.com/2024/04/21/material-testing-with-white-noise/
"""

import random

from scipy.stats import norm

import openseespy.opensees as ops

# Default test parameters (can be overridden via test_uniaxialMaterial)
_DEFAULT_NPULSE = 1000
_DEFAULT_NINCR = 10


def _build_modulated_noise(Npulse, seed=None):
    """Build normalized modulated white noise (one trapezoidal window)."""
    if seed is not None:
        random.seed(seed)
    noise = [norm.ppf(random.random()) for _ in range(Npulse)]

    def modulator(t, t1, t2, t3, t4):
        if t <= t1 or t >= t4:
            return 0.0
        if t < t2:
            return (t - t1) / (t2 - t1)
        if t < t3:
            return 1.0
        return 1.0 - (t - t3) / (t4 - t3)

    t1, t2 = 0, 0.1 * Npulse
    t3, t4 = 0.9 * Npulse, Npulse
    mod = [noise[i] * modulator(i, t1, t2, t3, t4) for i in range(Npulse)]
    scale = max(abs(max(mod)), abs(min(mod))) or 1.0
    return [x / scale for x in mod]


def test_uniaxialMaterial(
    *,
    material_tag,
    max_strain,
    Npulse=None,
    Nincr=None,
    seed=None,
):
    """
    Run a uniaxial material test with modulated white-noise strain history.

    Fixes the existing model, builds a side 1D model (two nodes, zeroLength
    element) with the given material, runs the analysis, then removes the side
    model and restores the original model to its previous state.

    Parameters
    ----------
    material_tag : int
        Tag of the uniaxialMaterial to test (in the current model).
    max_strain : float
        Peak strain magnitude applied to the normalized noise history.
    Npulse : int, optional
        Number of pulses in the noise history. Default 1000.
    Nincr : int, optional
        Analysis steps per pulse. Default 10.
    seed : int, optional
        Random seed for noise; if None, uses current RNG state.

    Returns
    -------
    eps : list[float]
        Imposed strain at each analysis step.
    sig : list[float]
        Stress from the material at each step.
    """
    Npulse = Npulse if Npulse is not None else _DEFAULT_NPULSE
    Nincr = Nincr if Nincr is not None else _DEFAULT_NINCR

    mod_noise = _build_modulated_noise(Npulse, seed=seed)
    Nsteps = Npulse * Nincr
    dt = 1.0 / Nincr

    # Back up current model and fix all existing nodes
    NDM = ops.getNDM()[0]
    NDF = ops.getNDF()[0]
    
    existing_node_tags = ops.getNodeTags()
    existing_constraints = {}
    fixed_nodes = ops.getFixedNodes()
    for nodeTag in existing_node_tags:
        if nodeTag in fixed_nodes:
            existing_constraints[nodeTag] = ops.getFixedDOFs(nodeTag)
            for dof in existing_constraints[nodeTag]:
                ops.remove("sp", nodeTag, dof)
    for nodeTag in existing_node_tags:
        fixities = [1] * NDF
        ops.fix(nodeTag, *fixities)

    # Build side model: 1D, 1 DOF
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    max_node_tag = max(existing_node_tags) if existing_node_tags else 0
    node1 = max_node_tag + 1
    node2 = max_node_tag + 2
    existing_ele_tags = ops.getEleTags()
    ele_tag = (max(existing_ele_tags) + 1) if existing_ele_tags else 1

    ops.node(node1, 0.0)
    ops.node(node2, 0.0)
    ops.fix(node1, 1)
    ops.element("zeroLength", ele_tag, node1, node2, "-mat", material_tag, "-dir", 1)

    ts_tag, pat_tag = ele_tag + 1, ele_tag + 2
    ops.timeSeries("Path", ts_tag, "-dt", 1, "-values", *mod_noise)
    ops.pattern("Plain", pat_tag, ts_tag, "-factor", max_strain)
    ops.sp(node2, 1, 1.0)

    # Reset domain time so Path is read from t=0 (needed when calling multiple times)
    ops.setTime(0.0)
    ops.wipeAnalysis()
    ops.constraints("Auto")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.integrator("LoadControl", dt)
    ops.analysis("Static", "-noWarnings")

    eps = []
    sig = []
    for _ in range(Nsteps):
        ops.analyze(1)
        sig.append(ops.eleResponse(ele_tag, "material", 1, "stress")[0])
        eps.append(ops.getLoadFactor(pat_tag))

    # Remove side model and restore original model
    ops.wipeAnalysis()
    ops.remove("sp", node1, 1)
    ops.remove("sp", node2, 1)
    ops.remove("loadPattern", pat_tag)
    ops.remove("timeSeries", ts_tag)
    ops.remove("element", ele_tag)
    ops.remove("node", node1)
    ops.remove("node", node2)
    ops.model("basic", "-ndm", NDM, "-ndf", NDF)
    for nodeTag in existing_node_tags:
        for dof in range(1, NDF + 1):
            try:
                ops.remove("sp", nodeTag, dof)
            except Exception:
                pass
    for nodeTag, constraints in existing_constraints.items():
        fixities = [0] * NDF
        for dof in constraints:
            fixities[dof - 1] = 1
        ops.fix(nodeTag, *fixities)

    # Reset domain time so caller starts from t=0 for next analysis
    ops.setTime(0.0)
    return eps, sig

