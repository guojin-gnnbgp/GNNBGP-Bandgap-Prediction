enable_testing()

find_program(BASH bash)
if(ENABLE_ASAN)
    set(PW_EXX_CPU_TEST_NAME 18_PW_EXX_CPU_test_with_asan)
    add_test(
        NAME ${PW_EXX_CPU_TEST_NAME}
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_CPU.txt -a ${ABACUS_BIN_PATH} -n 2 -s true -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
else()
    set(PW_EXX_CPU_TEST_NAME 18_PW_EXX_CPU)
    add_test(
        NAME ${PW_EXX_CPU_TEST_NAME}
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_CPU.txt -a ${ABACUS_BIN_PATH} -n 4 -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
    add_test(
        NAME 18_PW_EXX_CPU_SERIAL
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_CPU_SERIAL.txt -a ${ABACUS_BIN_PATH} -n 1 -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
endif()

if(ENABLE_HDF5)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME 18_PW_EXX_energy_density_HDF5_present
        COMMAND ${Python3_EXECUTABLE}
                ${ABACUS_TEST_DIR}/integrate/tools/validate_exx_energy_density_h5.py
                ${ABACUS_TEST_DIR}/18_PW_EXX/PW_PBE_training_exx_energy_density_gamma/OUT.autotest/training_dump/step_0.h5
                --expect-present
    )
    set_tests_properties(18_PW_EXX_energy_density_HDF5_present
                         PROPERTIES DEPENDS ${PW_EXX_CPU_TEST_NAME})

    add_test(
        NAME 18_PW_EXX_energy_density_HDF5_absent
        COMMAND ${Python3_EXECUTABLE}
                ${ABACUS_TEST_DIR}/integrate/tools/validate_exx_energy_density_h5.py
                ${ABACUS_TEST_DIR}/18_PW_EXX/PW_PBE_training_exx_label_auto/OUT.autotest/training_dump/step_0.h5
                --expect-absent
    )
    set_tests_properties(18_PW_EXX_energy_density_HDF5_absent
                         PROPERTIES DEPENDS ${PW_EXX_CPU_TEST_NAME})

    add_test(
        NAME 18_PW_EXX_band_edge_frontier_HDF5
        COMMAND ${Python3_EXECUTABLE}
                ${ABACUS_TEST_DIR}/integrate/tools/validate_band_edge_h5.py
                ${ABACUS_TEST_DIR}/18_PW_EXX/PW_PBE_training_band_edge_frontier_gamma/OUT.autotest/training_dump/step_0.h5
                --expect-band-only
    )
    set_tests_properties(18_PW_EXX_band_edge_frontier_HDF5
                         PROPERTIES DEPENDS ${PW_EXX_CPU_TEST_NAME})

    add_test(
        NAME 18_PW_EXX_band_edge_degenerate_HDF5
        COMMAND ${Python3_EXECUTABLE}
                ${ABACUS_TEST_DIR}/integrate/tools/validate_band_edge_h5.py
                ${ABACUS_TEST_DIR}/18_PW_EXX/PW_PBE_training_band_edge_degenerate_gamma/OUT.autotest/training_dump/step_0.h5
                --expect-band-only
                --expect-nedge 5
    )
    set_tests_properties(18_PW_EXX_band_edge_degenerate_HDF5
                         PROPERTIES DEPENDS ${PW_EXX_CPU_TEST_NAME})

endif()

add_test(
    NAME 18_PW_EXX_band_edge_frontier_omits_scalar_EXX
    COMMAND ${BASH} -c
            "! grep -Fq 'training_exx_label: full EXX energy' '${ABACUS_TEST_DIR}/18_PW_EXX/PW_PBE_training_band_edge_frontier_gamma/OUT.autotest/running_scf.log'"
)
set_tests_properties(18_PW_EXX_band_edge_frontier_omits_scalar_EXX
                     PROPERTIES DEPENDS ${PW_EXX_CPU_TEST_NAME})

if(USE_CUDA)
    add_test(
        NAME 18_PW_EXX_GPU
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_GPU.txt -a ${ABACUS_BIN_PATH} -n 4 -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
endif()
