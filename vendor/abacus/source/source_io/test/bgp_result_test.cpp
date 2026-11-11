#include "source_io/module_bgp/bgp_result.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace
{

std::string read_file(const std::string& path)
{
    std::ifstream input(path.c_str());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

class BgpResultTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        directory_ = "bgp_result_test_" + std::to_string(static_cast<long long>(getpid()));
        if (mkdir(directory_.c_str(), 0700) != 0 && errno != EEXIST)
        {
            throw std::runtime_error("could not create test directory");
        }
        path_ = directory_ + "/bgp_electronic_state.json";
    }

    void TearDown() override
    {
        std::remove(path_.c_str());
        std::remove((path_ + ".tmp").c_str());
        rmdir(directory_.c_str());
    }

    ModuleIO::BgpElectronicState valid_state(bool converged = true) const
    {
        ModuleIO::BgpElectronicState state;
        state.converged = converged;
        state.nspin = 1;
        state.fermi_energy_ry = 0.25;
        state.printed_gap_ry = 0.5;
        state.k_weights = {0.25, 0.75};
        state.eigenvalues_ry = {{-1.0, 1.0}, {-0.5, 1.5}};
        // ABACUS wg stores a full nspin=1 band as the k-point weight.
        state.occupations = {{0.25, 0.0}, {0.75, 0.0}};
        return state;
    }

    std::string directory_;
    std::string path_;
};

TEST_F(BgpResultTest, WritesVersionedElectronicStateInEv)
{
    ModuleIO::write_bgp_electronic_state(path_, valid_state(), 0);

    const std::string json = read_file(path_);
    EXPECT_NE(json.find("\"schema_version\": \"1.0\""), std::string::npos);
    EXPECT_NE(json.find("\"converged\": true"), std::string::npos);
    EXPECT_NE(json.find("\"nspin\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"occupation_weighting\": \"abacus_k_weighted\""), std::string::npos);
    EXPECT_NE(json.find("\"spin_capacity\": 2.0"), std::string::npos);
    EXPECT_NE(json.find("\"energy_unit\": \"eV\""), std::string::npos);
    EXPECT_NE(json.find("\"fermi_energy_ev\": 3.4014245"), std::string::npos);
    EXPECT_NE(json.find("\"printed_gap_ev\": 6.802849"), std::string::npos);
    EXPECT_NE(json.find("\"k_weights\": [0.25, 0.75]"), std::string::npos);
    EXPECT_NE(json.find("\"eigenvalues_ev\": [[-13.605698, 13.605698], [-6.802849, 20.408547]]"),
              std::string::npos);
    EXPECT_NE(json.find("\"occupations\": [[0.25, 0], [0.75, 0]]"), std::string::npos);
}

TEST_F(BgpResultTest, WritesUnconvergedStateAndAtomicallyReplacesExistingFile)
{
    {
        std::ofstream existing(path_.c_str());
        existing << "stale";
    }

    ModuleIO::write_bgp_electronic_state(path_, valid_state(false), 0);

    const std::string json = read_file(path_);
    EXPECT_NE(json.find("\"converged\": false"), std::string::npos);
    EXPECT_EQ(json.find("stale"), std::string::npos);
    EXPECT_EQ(access((path_ + ".tmp").c_str(), F_OK), -1);
}

TEST_F(BgpResultTest, DoesNothingAwayFromRankZero)
{
    ModuleIO::write_bgp_electronic_state(path_, valid_state(), 1);
    EXPECT_EQ(access(path_.c_str(), F_OK), -1);
}

TEST_F(BgpResultTest, RejectsUnsupportedSpinDimensionsAndNonFiniteValues)
{
    auto state = valid_state();
    state.nspin = 2;
    EXPECT_THROW(ModuleIO::write_bgp_electronic_state(path_, state, 0), std::runtime_error);

    state = valid_state();
    state.eigenvalues_ry.pop_back();
    EXPECT_THROW(ModuleIO::write_bgp_electronic_state(path_, state, 0), std::runtime_error);

    state = valid_state();
    state.occupations[0].pop_back();
    EXPECT_THROW(ModuleIO::write_bgp_electronic_state(path_, state, 0), std::runtime_error);

    state = valid_state();
    state.k_weights[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(ModuleIO::write_bgp_electronic_state(path_, state, 0), std::runtime_error);
}

} // namespace
