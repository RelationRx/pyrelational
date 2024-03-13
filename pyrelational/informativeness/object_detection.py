"""
This module contains methods for scoring samples based on model uncertainty in
object detection tasks

This module contains functions for computing the informativeness values
of a given probability distribution (outputs of a model/mc-dropout
prediction, etc.)
"""

import math

import torch
from torch import Tensor


def object_detection_least_confidence(
    prob_dist: Tensor, axis: int = -1, aggregation_type: str = "max"
) -> Tensor:
    r"""Returns the informativeness score of an array using least confidence
    sampling in a 0-1 range where 1 is the most uncertain

    The least confidence uncertainty is the normalised difference between
    the most confident prediction and 100 percent confidence

    :param prob_dist: real number tensor whose elements add to 1.0 along an axis
    :param axis: axis of prob_dist where probabilities add to 1

    :return: tensor with normalised least confidence scores

    """
    # assert torch.allclose(
    #     prob_dist.sum(axis), torch.tensor(1.0)
    # ), "input should be probability distributions along specified axis"
    uncertainty = []
    for batch_conf_scores in prob_dist:
        for img_conf_scores in batch_conf_scores:
            img_least_conf_score = []
            for bbox_scores in img_conf_scores:
                bbox_conf_scores = softmax(bbox_scores)
                bbox_least_conf_score: Tensor = 1 - torch.max(bbox_conf_scores)
                img_least_conf_score.append(bbox_least_conf_score)

            uncertainty.append(
                compute_total_uncertainty(
                    img_least_conf_score, aggregation_type
                )
            )
    return uncertainty


def object_detection_margin_confidence(
    prob_dist: Tensor, axis: int = -1, aggregation_type: str = "max"
) -> Tensor:
    r"""Returns the informativeness score of a probability distribution using
    margin of confidence sampling in a 0-1 range where 1 is the most uncertain
    The margin confidence uncertainty is the difference between the top two
    most confident predictions

    :param prob_dist: real number tensor whose elements add to 1.0 along an axis
    :param axis: axis of prob_dist where probabilities add to 1

    :return: tensor with margin confidence scores
    """
    uncertainty = []
    for batch_conf_scores in prob_dist:
        for img_conf_scores in batch_conf_scores:
            img_margin_conf_score = []
            for bbox_scores in img_conf_scores:
                bbox_conf_scores = softmax(bbox_scores)
                bbox_conf_scores, _ = torch.sort(
                    bbox_conf_scores, descending=True, dim=axis
                )
                difference = bbox_conf_scores.select(
                    axis, 0
                ) - bbox_conf_scores.select(axis, 1)

                bbox_margin_conf_score: Tensor = 1 - difference

                img_margin_conf_score.append(bbox_margin_conf_score)

            uncertainty.append(
                compute_total_uncertainty(
                    img_margin_conf_score, aggregation_type
                )
            )

    return uncertainty


def object_detection_entropy(
    prob_dist: Tensor, axis: int = -1, aggregation_type: str = "max"
) -> Tensor:
    r"""Returns the informativeness score of a probability distribution
    using entropy

    The entropy based uncertainty is defined as

    :math:`- \frac{1}{\log(n)} \sum_{i}^{n} p_i \log (p_i)`

    :param prob_dist: real number tensor whose elements add to 1.0 along an axis
    :param axis: axis of prob_dist where probabilities add to 1

    :return: tensor of entropy based uncertainties
    """
    # assert torch.allclose(
    #     prob_dist.sum(axis), torch.tensor(1.0)
    # ), "input should be probability distributions along specified axis"

    uncertainty = []
    for batch_conf_scores in prob_dist:
        for img_conf_scores in batch_conf_scores:
            img_entropy = []
            for bbox_scores in img_conf_scores:
                bbox_conf_scores = softmax(bbox_scores)
                bbox_entropy_score = -torch.sum(
                    bbox_conf_scores * torch.log2(bbox_conf_scores)
                )
                img_entropy.append(bbox_entropy_score)

            uncertainty.append(
                compute_total_uncertainty(img_entropy, aggregation_type)
            )
    return uncertainty

    # log_probs = prob_dist * torch.log2(prob_dist)
    # raw_entropy = 0 - torch.sum(log_probs, dim=axis)
    # normalised_entropy: Tensor = raw_entropy / math.log2(
    #     prob_dist.size(axis)
    # )
    # return normalised_entropy


def softmax(scores: Tensor, base: float = math.e, axis: int = -1) -> Tensor:
    """Returns softmax array for array of scores

    Converts a set of raw scores from a model (logits) into a
    probability distribution via softmax.

    The probability distribution will be a set of real numbers
    such that each is in the range 0-1.0 and the sum is 1.0.

    Assumes input is a pytorch tensor: tensor([1.0, 4.0, 2.0, 3.0])

    :param scores: (pytorch tensor) a pytorch tensor of any positive/negative real numbers.
    :param base: the base for the exponential (default e)
    :param: axis to apply softmax on scores

    :return: tensor of softmaxed scores
    """
    exps = base ** scores.float()  # exponential for each value in array
    sum_exps = torch.sum(
        exps, dim=axis, keepdim=True
    )  # sum of all exponentials
    prob_dist: Tensor = exps / sum_exps  # normalize exponentials
    return prob_dist


def compute_total_uncertainty(img_uncertainty: list, aggregation_type: str):
    """Helper function that computes a value metric to deal with uncetainties in an image,
    since each image has differnet number of bounding boxes.

    :param img_uncertainty: list containing the uncertainty of each bounding box in an image.
    :param aggregation_type: function to handle with many uncetrainty values in an image.

    :return: Final uncertainty score for the whole image.
    """
    if len(img_uncertainty) == 0:
        final_uncertainty_score = 0
    else:
        img_entropy_torch = torch.stack(img_uncertainty)
        if aggregation_type == "max":
            final_uncertainty_score = max(img_uncertainty)
        elif aggregation_type == "L2":
            final_uncertainty_score = torch.norm(img_entropy_torch)
        else:
            raise ValueError(
                f'Aggregation type "{aggregation_type}" not recognise'
            )
    return final_uncertainty_score
