// 융합전자공학부 220125120 김태우
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#define SIZE 100

int main(void)
{
    int integer[SIZE];
    int i, k;

    for (i = 0; i < SIZE; i++)
    {
        printf("정수를 입력하시오: ");
        scanf("%d", integer[i]);

        if (integer[i] == -1)
        {
            break;
        }
    }

    for (k = 0; k < SIZE; k++)
    {
        for (i = 0; i < SIZE - 1; i++)
        {

            if (integer[i] > integer[i + 1])
            {
                int tmp = integer[i];
                integer[i] = integer[i + 1];
                integer[i + 1] = tmp;
            }
        }
    }
    return 0;
}