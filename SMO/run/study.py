import sys

sys.path.insert(0, "../..")
from SMO.factories import CaseFactory
from SMO.utils import data_to_be_used
import numpy as np
import torch
from copy import deepcopy
import uuid
import time
from SMO.analysis import Log
from SMO.optimization.objectives import WrapperModel
from SMO.optimization.em import ExpectationMaximization
from SMO.optimization.active import setup_active_learner
from SMO.optimization.em import AdaptiveConvergenceCriteria


def create_surrgate(factory):
    return factory.discriminative()


def setup(
    identifier, n_data, fargs, cargs, targs, init, train=True, objective_fct=None
):

    T1 = time.time()

    factory = CaseFactory.FromIdentifier(identifier, fargs)
    rf, rfp = factory.rf()
    rf.set_phi(deepcopy(init["phi"]))
    dtype = factory.dtype
    device = factory.device
    dtransform = factory.dtransform()

    data_tr, data_val = factory.data(
        N_training=n_data,
        N_validation=cargs["N_validation"],
        permute_returned_data=True,
        permutation_tr=init["permutation_tr"],
    )

    with torch.no_grad():
        data_tr["X_cnn"] = dtransform(data_tr["X_g"])
        data_val["X_cnn"] = dtransform(data_val["X_g"])

    surrogate = create_surrgate(factory)
    wmodel = WrapperModel(
        surrogate,
        rf,
        rfp,
        dtransform,
        target=factory.target,
        hmg=factory.hmg(),
        htransform=factory.htransform(),
    ).to(dtype=dtype, device=device)
    wmodel.eval()

    training_params = deepcopy(targs)
    training_params["init_dict"] = deepcopy(init["surrogate_params"])
    training_params["data_val"] = data_val
    wmodel.autotrain(training_params)

    if objective_fct is None:
        objective = factory.objective(wmodel, cargs, dtype=dtype, device=device)
    else:
        objective = objective_fct(wmodel, cargs, dtype=dtype, device=device)

    active, dhandler = setup_active_learner(
        factory,
        data_tr,
        data_val,
        wmodel,
        objective,
        cargs["N_data_acquisitions"],
        cargs["N_add"],
        cargs["N_candidates"],
        learning_strategy=cargs["active_strategy"],
    )
    em = ExpectationMaximization(objective)

    if train:
        trainer = wmodel.autotrain()
    else:
        trainer = None

    return (
        factory,
        rf,
        em,
        wmodel,
        trainer,
        objective,
        active,
        dhandler,
        T1,
        dtype,
        device,
    )


def study_active(identifier, folder, iteration, fargs, cargs, targs, init, descriptor):

    print("======================= ACTIVE LEARNING =============================")

    log = Log(identifier, descriptor)
    log.data["N_training_sequence"] = data_to_be_used(cargs)
    log.data["shared_uuid"] = deepcopy(init["shared_uuid"])
    log.data["phi0"] = deepcopy(init["phi"])

    (
        factory,
        rf,
        em,
        wmodel,
        trainer,
        objective,
        active,
        dhandler,
        T1,
        dtype,
        device,
    ) = setup(identifier, cargs["N_training_init"], fargs, cargs, targs, init)

    t1 = time.time()
    ConvergenceCriteria = AdaptiveConvergenceCriteria(
        patience=cargs["patience"],
        num_max_steps=cargs["N_em_max_steps"],
        cooldown=cargs["cooldown"],
        verbose=True,
    )
    elbos = em.run_active_learning(
        ConvergenceCriteria,
        active,
        N_elbo_estimate=cargs["N_monte_carlo_elbo"],
        N_objective_estimate=cargs["N_objective_fct_monte_carlo"],
        callback=None,
        log=log,
        M_steps=cargs["M_steps"],
    )
    t2 = time.time()

    log.data["trainingdata"] = dhandler.export(clone=True)
    log.data["dargs"] = wmodel._discriminative.options
    log.touch()
    log.save(folder)


def study_baseline(
    identifier, folder, iteration, fargs, cargs, targs, init, descriptor
):

    print("======================= BASELINE =============================")

    if "N_training_baseline" in cargs and cargs["N_training_baseline"]:
        N_training = cargs["N_training_baseline"]
    else:
        N_training = [
            cargs["N_training_init"] + n * cargs["N_add"]
            for n in range(cargs["N_data_acquisitions"] + 1)
        ]

    log = Log(identifier, descriptor)
    log.data["N_training_sequence"] = N_training
    log.data["shared_uuid"] = deepcopy(init["shared_uuid"])
    log.data["phi0"] = deepcopy(init["phi"])

    print("Data sequence used: {}".format(N_training))

    for n_data in N_training:

        state = log.add_state(n_data)
        (
            factory,
            rf,
            em,
            wmodel,
            trainer,
            objective,
            active,
            dhandler,
            T1,
            dtype,
            device,
        ) = setup(identifier, n_data, fargs, cargs, targs, init)
        assert (
            max(N_training) <= factory.N_training_available
        ), "The precomputed data cannot satisfy the request"

        ConvergenceCriteria = AdaptiveConvergenceCriteria(
            patience=cargs["patience"],
            num_max_steps=cargs["N_em_max_steps"],
            cooldown=cargs["cooldown"],
            verbose=True,
        )
        elbos = em.run(
            ConvergenceCriteria,
            N_elbo_estimate=cargs["N_monte_carlo_elbo"],
            M_steps=cargs["M_steps"],
        )
        _, _, kappa_sample_base = active._homogenize(
            torch.randn(
                cargs["N_objective_fct_monte_carlo"],
                rf.kernel.dim_phase_angles,
                dtype=dtype,
                device=device,
            )
        )
        oval_base = objective.assess(kappa_sample_base)

        T2 = time.time()

        state.inform(em, active=None)
        state["objective"] = oval_base
        state._kappa_samples_ref = kappa_sample_base.detach().cpu()

    log.data["dargs"] = wmodel._discriminative.options
    log.touch()
    log.save(folder)


def make_init(factory, device, dtype=None):

    assert isinstance(device, torch.device)

    init = dict()
    init["phi"] = np.random.normal(0, 1, factory.pdim)
    init["surrogate_params"] = deepcopy(factory.discriminative().state())  #
    init["permutation_tr"] = torch.randperm(factory.N_training_available).to(
        device=device
    )
    init["shared_uuid"] = str(uuid.uuid4())

    return init
