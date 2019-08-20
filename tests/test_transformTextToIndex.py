from unittest import TestCase

from transform_text_to_index import TransformTextToIndex


class TestTransformTextToIndex(TestCase):
    def test_transform(self):
        max_feature_len = [5, 10]
        sut = TransformTextToIndex(max_feature_len)

        input_data = [
            # batch size = 3
            [
                # bx
                [
                    # columns 2 features
                    # col 1
                    [
                        # 3 Rows within columns
                        'test1@yahoo.com'
                        , 'test2@yahoo.com'
                        , 'test22@yahoo.com'
                    ]
                    ,
                    # col 2
                    [
                        "sample data"
                        , "Sample data2"
                        , "Sample data3"
                    ]

                ],

                # by
                ["NO", "NO", "YES"]
            ]

        ]

        expected = [
            # b
            [
                # bx
                [
                    # c1
                    [
                        # 3 rows of character to index all fixed to same length
                        [29, 14, 28, 29, 1],
                        [29, 14, 28, 29, 2],
                        [29, 14, 28, 29, 2]
                    ],
                    # c2
                    [
                        # 3 rows
                        [28, 10, 22, 25, 21, 14, 94, 13, 10, 29],
                        [54, 10, 22, 25, 21, 14, 94, 13, 10, 29],
                        [54, 10, 22, 25, 21, 14, 94, 13, 10, 29]
                    ]
                ],
                # by
                ['NO', 'NO', 'YES']
            ]
        ]

        # Act
        actual = sut.transform(input_data)

        # Assert
        self.assertEqual(expected, actual)
