#ifndef MODULE_HAMILT_CIDER_TEST_ORACLE_SHA256_H
#define MODULE_HAMILT_CIDER_TEST_ORACLE_SHA256_H

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace cider_test
{
namespace detail
{

inline std::uint32_t rotate_right(std::uint32_t value, unsigned int count)
{
    return (value >> count) | (value << (32U - count));
}

inline std::string sha256(const std::vector<std::uint8_t>& input)
{
    static const std::uint32_t constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

    std::vector<std::uint8_t> message = input;
    const std::uint64_t bit_count = static_cast<std::uint64_t>(message.size()) * 8U;
    message.push_back(0x80U);
    while (message.size() % 64U != 56U)
    {
        message.push_back(0U);
    }
    for (int shift = 56; shift >= 0; shift -= 8)
    {
        message.push_back(static_cast<std::uint8_t>(bit_count >> shift));
    }

    std::uint32_t state[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                              0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                              0x1f83d9abU, 0x5be0cd19U};
    for (std::size_t chunk = 0; chunk < message.size(); chunk += 64U)
    {
        std::uint32_t words[64] = {};
        for (std::size_t index = 0; index < 16U; ++index)
        {
            const std::size_t offset = chunk + 4U * index;
            words[index] = (static_cast<std::uint32_t>(message[offset]) << 24U)
                           | (static_cast<std::uint32_t>(message[offset + 1U]) << 16U)
                           | (static_cast<std::uint32_t>(message[offset + 2U]) << 8U)
                           | static_cast<std::uint32_t>(message[offset + 3U]);
        }
        for (std::size_t index = 16U; index < 64U; ++index)
        {
            const std::uint32_t s0 = rotate_right(words[index - 15U], 7U)
                                     ^ rotate_right(words[index - 15U], 18U)
                                     ^ (words[index - 15U] >> 3U);
            const std::uint32_t s1 = rotate_right(words[index - 2U], 17U)
                                     ^ rotate_right(words[index - 2U], 19U)
                                     ^ (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
        }

        std::uint32_t a = state[0];
        std::uint32_t b = state[1];
        std::uint32_t c = state[2];
        std::uint32_t d = state[3];
        std::uint32_t e = state[4];
        std::uint32_t f = state[5];
        std::uint32_t g = state[6];
        std::uint32_t h = state[7];
        for (std::size_t index = 0; index < 64U; ++index)
        {
            const std::uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U)
                                       ^ rotate_right(e, 25U);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1
                = h + sum1 + choose + constants[index] + words[index];
            const std::uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U)
                                       ^ rotate_right(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (std::uint32_t word : state)
    {
        result << std::setw(8) << word;
    }
    return result.str();
}

} // namespace detail

inline std::string sha256_little_endian_doubles(const std::vector<double>& values)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(values.size() * sizeof(double));
    for (double value : values)
    {
        std::uint64_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int shift = 0U; shift < 64U; shift += 8U)
        {
            bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
        }
    }
    return detail::sha256(bytes);
}

} // namespace cider_test

#endif
