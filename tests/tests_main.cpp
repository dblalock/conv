#include <gtest/gtest.h>

TEST(Main, Smoketest) {
    auto x = 5;
    EXPECT_EQ(x, 5);
    EXPECT_GT(x, 4);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    printf("test_main done\n");
    return ret;
}
