import itertools
import math
import os
import random
from time import perf_counter

from vimms.Box import GenericBox, DictGrid, ArrayGrid, LocatorGrid, \
    AllOverlapGrid, IdentityDrift
from vimms.ChemicalSamplers import DatabaseFormulaSampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import POSITIVE, load_obj, set_log_level_warning
from vimms.Controller.box import NonOverlapController
from vimms.Environment import Environment
from vimms.GridEstimator import GridEstimator
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Noise import GaussianPeakNoise


class BoxEnv():
    def __init__(self, min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen,
                 max_ylen):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.max_mz = max_mz
        self.min_x1, self.max_x1 = min_rt, max_rt - max_xlen
        self.min_xlen, self.max_xlen, self.min_ylen, self.max_ylen = \
            min_xlen, max_xlen, min_ylen, max_ylen
        self.grid = None

    def init_grid(self, grid_class, rt_box_size, mz_box_size):
        self.grid = grid_class(self.min_rt, self.max_rt, rt_box_size, 0,
                               self.max_mz, mz_box_size)

    def generate_box(self):
        x1 = random.uniform(self.min_x1, self.max_x1)
        y1 = random.uniform(0, self.max_mz - self.max_ylen)
        xlen = random.uniform(self.min_xlen, self.max_xlen)
        ylen = random.uniform(self.min_ylen, self.max_ylen)
        return GenericBox(x1, x1 + xlen, y1, y1 + ylen, intensity=1)

    @classmethod
    def random_boxenv(cls):
        min_rt, max_rt = 0, random.randint(1000, 2000)
        max_mz = random.randint(1000, 3000)
        min_xlen = random.randint(1, 4)
        max_xlen = random.randint(min_xlen, 10)
        min_ylen = random.randint(1, 5)
        max_ylen = random.randint(min_ylen, 10)
        return BoxEnv(min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen,
                      max_ylen)

    def box_score(self, box): return self.grid.non_overlap(box)

    def register_box(self, box): self.grid.register_box(box)


class TestEnv(BoxEnv):
    def __init__(self, min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen,
                 max_ylen):
        super().__init__(min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen,
                         max_ylen)
        self.boxes_by_injection = [[]]

    @classmethod
    def random_boxenv(cls, boxes_per_injection, no_injections):
        boxenv = super().random_boxenv()
        boxenv = TestEnv(boxenv.min_rt, boxenv.max_rt, boxenv.max_mz,
                         boxenv.min_xlen, boxenv.max_xlen, boxenv.min_ylen,
                         boxenv.max_ylen)
        boxenv.boxes_by_injection = [[boxenv.generate_box()
                                      for j in range(boxes_per_injection)]
                                     for i in range(no_injections)]
        return boxenv

    def test_simple_splitter(self):
        return [
            [LocatorGrid.splitting_non_overlap(box, itertools.chain(
                *self.boxes_by_injection[:i], inj[:j])) for j, box
             in enumerate(inj)] for i, inj in
            enumerate(self.boxes_by_injection)]

    def test_non_overlap(self, grid_class, rt_box_size, mz_box_size):
        self.init_grid(grid_class, rt_box_size, mz_box_size)

        def score_box(box):
            score = self.grid.non_overlap(box)
            self.grid.register_box(box)
            return score

        return [[score_box(b) for b in inj] for inj in self.boxes_by_injection]

    def test_intensity_non_overlap(self, grid_class, rt_box_size, mz_box_size):
        self.init_grid(grid_class, rt_box_size, mz_box_size)

        def score_box(box):
            score = self.grid.intensity_non_overlap(box, box.intensity,
                                                    {"theta1": 1})
            self.grid.register_box(box)
            return score

        return [[score_box(b) for b in inj] for inj in self.boxes_by_injection]


def run_vimms(no_injections, rt_box_size, mz_box_size):
    rt_range = [(0, 1440)]
    min_rt, max_rt = rt_range[0]
    ionisation_mode, isolation_width = POSITIVE, 1
    N, rt_tol, mz_tol, min_ms1_intensity = 10, 15, 10, 5000
    min_roi_intensity, min_roi_length, min_roi_length_for_fragmentation = \
        500, 3, 3
    grid = GridEstimator(
        LocatorGrid(min_rt, max_rt, rt_box_size, 0, 3000, mz_box_size),
        IdentityDrift())

    hmdbpath = os.path.join(os.path.abspath(os.getcwd()), "..", "..", "tests",
                            "fixtures", "hmdb_compounds.p")
    hmdb = load_obj(hmdbpath)
    df = DatabaseFormulaSampler(hmdb, min_mz=100, max_mz=1000)
    cm = ChemicalMixtureCreator(df, adduct_prior_dict={POSITIVE: {"M+H": 1}})
    chemicals = cm.sample(2000, 1)

    boxes = []
    for i in range(no_injections):
        mz_noise = GaussianPeakNoise(0.1)
        mass_spec = IndependentMassSpectrometer(POSITIVE, chemicals,
                                                mz_noise=mz_noise)
        controller = NonOverlapController(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
            min_roi_intensity,
            min_roi_length, N, grid, rt_tol=rt_tol,
            min_roi_length_for_fragmentation=min_roi_length_for_fragmentation
        )
        env = Environment(mass_spec, controller, min_rt, max_rt,
                          progress_bar=True)
        set_log_level_warning()
        env.run()
        boxes.append(
            [r.to_box(0.01, 0.01) for r in controller.roi_builder.get_rois()])
    return boxes


# flake8: noqa: C901
def main():
    class Timer():
        def __init__(self): self.time = None

        def start_time(self): self.time = perf_counter()

        def end_time(self): return perf_counter() - self.time

        def time_f(self, f):
            self.start_time()
            result = f()
            return result, self.end_time()

    def run_area_calcs(boxenv, rt_box_size, mz_box_size):
        def pretty_print(scores):
            print({i: x for i, x in enumerate(itertools.chain(*scores)) if
                   x > 0.0 and x < 1.0})

        print("\nRun area calcs start!")
        print("\nDictGrid Scores:")
        scores_by_injection, dict_time = Timer().time_f(
            lambda: boxenv.test_non_overlap(
                DictGrid, rt_box_size, mz_box_size))
        pretty_print(scores_by_injection)

        print("\nBoolArrayGrid Scores:")
        scores_by_injection_2, array_time = Timer().time_f(
            lambda: boxenv.test_non_overlap(ArrayGrid, rt_box_size,
                                            mz_box_size))
        pretty_print(scores_by_injection_2)

        print("\nExact Scores:")
        scores_by_injection_3, exact_time = Timer().time_f(
            lambda: boxenv.test_simple_splitter())
        pretty_print(scores_by_injection_3)

        print("\nExact Scores Grid:")
        rt_box_size, mz_box_size = (
                                           boxenv.max_rt - boxenv.min_rt) / 50, boxenv.max_mz / 50
        scores_by_injection_4, exact_grid_time = Timer().time_f(
            lambda: boxenv.test_non_overlap(LocatorGrid, rt_box_size,
                                            mz_box_size))
        pretty_print(scores_by_injection_4)

        def compare_scores(scores_1, scores_2):
            return {i: (x, y) for i, (x, y) in enumerate(
                zip(itertools.chain(*scores_1), itertools.chain(*scores_2))) if
                    not math.isclose(x, y)}

        print("Differences between grid + no grid:",
              compare_scores(scores_by_injection_3, scores_by_injection_4))
        # note: below non_overlap (not multiplied by intensity) +
        # intensity_non_overlap should have same behaviour assuming that
        # all box intensities are 1
        print("Differences between no intensity and intensity overlap:",
              compare_scores(scores_by_injection_4,
                             boxenv.test_intensity_non_overlap(
                                 AllOverlapGrid, rt_box_size,
                                 mz_box_size)))

        print("\nDictGrid Time Taken: {}".format(dict_time))
        print("BoolArray Time Taken: {}".format(array_time))
        print("BoxSplitting Time Taken: {}".format(exact_time))
        print("BoxSplitting with Grid Time Taken {}".format(exact_grid_time))

    def box_adjust(boxenv, *no_boxes):
        for x_n, y_n in no_boxes:
            rt_box_size, mz_box_size = (
                                               boxenv.max_rt - boxenv.min_rt) / x_n, boxenv.max_mz / y_n
            _, exact_grid_time = Timer().time_f(
                lambda: boxenv.test_non_overlap(LocatorGrid, rt_box_size,
                                                mz_box_size))
            print(
                "Time with {}, {} Boxes: {}".format(x_n, y_n, exact_grid_time))

    boxenv = TestEnv.random_boxenv(200, 3)
    run_area_calcs(boxenv, (boxenv.max_rt - boxenv.min_rt) / 10000,
                   boxenv.max_mz / 10000)

    boxenv = TestEnv(0, 50, 50, 2, 3, 2, 3)
    boxenv.boxes_by_injection = [[GenericBox(0, 10, 0, 30, intensity=1),
                                  GenericBox(5, 15, 0, 30, intensity=2),
                                  GenericBox(0, 10, 15, 45, intensity=3),
                                  GenericBox(0, 17, 0, 30, intensity=4)]]
    run_area_calcs(boxenv, 0.2, 0.2)
    print("Intensity Non-Overlap Scores: ",
          boxenv.test_intensity_non_overlap(AllOverlapGrid, 0.2, 0.2))

    print()

    box = GenericBox(0, 10, 0, 10)
    other_boxes = [[GenericBox(0 + x, 10 + x, 0, 10) for x in range(0, 11)],
                   [GenericBox(0, 10, 0 + y, 10 + y) for y in range(0, 11)],
                   [GenericBox(0 + n, 10 + n, 0 + n, 10 + n) for n in
                    range(0, 11)]]
    for ls in other_boxes:
        print([box.overlap_2(b) for b in ls])

    print()

    boxenv = TestEnv(0, 1440, 1500, 0, 0, 0, 0)
    vimms_boxes = run_vimms(20, (boxenv.max_rt - boxenv.min_rt) / 150,
                            boxenv.max_mz / 150)
    boxenv.boxes_by_injection = vimms_boxes
    run_area_calcs(boxenv, 0.2, 0.01)

    print()
    for ratio in range(1, 11):
        print("---Ratio of {}---\n".format(ratio))
        box_adjust(boxenv,
                   *((n // ratio, n) for n in range(ratio, 1001, 10 * ratio)))

    from statistics import mean

    def box_lengths(b):
        return b.pt2.x - b.pt1.x, b.pt2.y - b.pt1.y

    print("Avg. xlen == {}, Avg. ylen == {}".format(
        *map(mean,
             zip(*(box_lengths(b) for inj in boxenv.boxes_by_injection for b in
                   inj)))))

    boxenv = TestEnv(0, 1440, 1500, 0, 0, 0, 0)
    boxenv.boxes_by_injection = vimms_boxes
    grid = AllOverlapGrid(0, 2000, 100, 0, 3000, 100)
    _, time = Timer().time_f(lambda: grid.boxes_by_overlaps(
        boxes=itertools.chain(*boxenv.boxes_by_injection)))
    print(f"Time taken for split all no grid: {time}")

    def split_all():
        for b in itertools.chain(*boxenv.boxes_by_injection):
            grid.register_box(
                b)
        return grid.boxes_by_overlaps()

    _, time = Timer().time_f(split_all)
    print(f"Time taken for split all grid: {time}")


if __name__ == "__main__":
    main()
