from lib.Free_energy_classes import Free_energy_contribution
from pathlib import Path

path_cwd = Path.cwd()

for mat in ['CsPbI3', 'FAPbI3', 'MAPbI3']:
    Path_mat = path_cwd / mat
    for phase in ['gamma', 'Csdelta', 'FAdelta']:
        Path_phase = Path_mat / phase

        Path_REX = Path_phase / 'REX_NPT'

        for T_NVT in [150, 187, 234, 293, 350, 419, 501, 600]:
            Path_NVT = Path_phase / str('NVT_T' + str(T_NVT)) / 'NPT_MD'

            fec_ref = Free_energy_contribution.from_p(Path_NVT / 'fec.pickle')
            fec_NPT = Free_energy_contribution.from_p(Path_REX / 'fec.pickle')

            fec_NPT.calculate_prop(T_NVT, fec_ref.free_energy)
            fec_NPT.calculate_error_prop(T_NVT, A_ref_error=fec_ref.free_error)

            fec_NPT.write_pickle_file(Path_REX / str("fec_ref_T" + str(T_NVT) +".pickle"))

        Path_REX = Path_phase / 'REX_NPT_sup'

        if mat == 'CsPbI3':
            T_lst_sup = [150, 350, 600]
        else:
            T_lst_sup = [234, 350, 501]

        for T_NVT in T_lst_sup:
            Path_NVT = Path_phase / str('NVT_T' + str(T_NVT)) / 'NPT_MD_sup'

            fec_ref = Free_energy_contribution.from_p(Path_NVT / 'fec.pickle')
            fec_NPT = Free_energy_contribution.from_p(Path_REX / 'fec.pickle')

            fec_NPT.calculate_prop(T_NVT, fec_ref.free_energy)
            fec_NPT.calculate_error_prop(T_NVT, A_ref_error=fec_ref.free_error)

            fec_NPT.write_pickle_file(Path_REX / str("fec_ref_T" + str(T_NVT) +".pickle"))
