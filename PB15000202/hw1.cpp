#include "SubImageMatch.h"


#define IMG_SHOW

#define MY_OK 1
#define MY_FAIL -1
#ifdef IMG_SHOW
#define TEST_TIMES 1
#endif
#ifndef IMG_SHOW
#define TEST_TIMES 5
#endif
using namespace cv;
using namespace std;


const float atan2buf[] = {
	0x00000000,0x00000065,0x000000C9,0x0000012E,0x00000192,0x000001F7,0x0000025B,0x000002C0,
	0x00000324,0x00000389,0x000003ED,0x00000452,0x000004B7,0x0000051B,0x00000580,0x000005E4,
	0x00000649,0x000006AD,0x00000712,0x00000777,0x000007DB,0x00000840,0x000008A5,0x00000909,
	0x0000096E,0x000009D3,0x00000A37,0x00000A9C,0x00000B01,0x00000B65,0x00000BCA,0x00000C2F,
	0x00000C94,0x00000CF8,0x00000D5D,0x00000DC2,0x00000E27,0x00000E8C,0x00000EF1,0x00000F55,
	0x00000FBA,0x0000101F,0x00001084,0x000010E9,0x0000114E,0x000011B3,0x00001218,0x0000127D,
	0x000012E2,0x00001347,0x000013AC,0x00001412,0x00001477,0x000014DC,0x00001541,0x000015A6,
	0x0000160C,0x00001671,0x000016D6,0x0000173C,0x000017A1,0x00001806,0x0000186C,0x000018D1,
	0x00001937,0x0000199C,0x00001A02,0x00001A67,0x00001ACD,0x00001B33,0x00001B98,0x00001BFE,
	0x00001C64,0x00001CCA,0x00001D2F,0x00001D95,0x00001DFB,0x00001E61,0x00001EC7,0x00001F2D,
	0x00001F93,0x00001FF9,0x0000205F,0x000020C5,0x0000212C,0x00002192,0x000021F8,0x0000225E,
	0x000022C5,0x0000232B,0x00002392,0x000023F8,0x0000245F,0x000024C5,0x0000252C,0x00002593,
	0x000025F9,0x00002660,0x000026C7,0x0000272E,0x00002795,0x000027FC,0x00002863,0x000028CA,
	0x00002931,0x00002998,0x000029FF,0x00002A66,0x00002ACE,0x00002B35,0x00002B9D,0x00002C04,
	0x00002C6C,0x00002CD3,0x00002D3B,0x00002DA2,0x00002E0A,0x00002E72,0x00002EDA,0x00002F42,
	0x00002FAA,0x00003012,0x0000307A,0x000030E2,0x0000314A,0x000031B3,0x0000321B,0x00003283,
	0x000032EC,0x00003354,0x000033BD,0x00003426,0x0000348E,0x000034F7,0x00003560,0x000035C9,
	0x00003632,0x0000369B,0x00003704,0x0000376D,0x000037D7,0x00003840,0x000038AA,0x00003913,
	0x0000397D,0x000039E6,0x00003A50,0x00003ABA,0x00003B24,0x00003B8E,0x00003BF8,0x00003C62,
	0x00003CCC,0x00003D36,0x00003DA0,0x00003E0B,0x00003E75,0x00003EE0,0x00003F4A,0x00003FB5,
	0x00004020,0x0000408B,0x000040F6,0x00004161,0x000041CC,0x00004237,0x000042A2,0x0000430E,
	0x00004379,0x000043E5,0x00004451,0x000044BC,0x00004528,0x00004594,0x00004600,0x0000466C,
	0x000046D8,0x00004745,0x000047B1,0x0000481D,0x0000488A,0x000048F7,0x00004963,0x000049D0,
	0x00004A3D,0x00004AAA,0x00004B17,0x00004B84,0x00004BF2,0x00004C5F,0x00004CCD,0x00004D3A,
	0x00004DA8,0x00004E16,0x00004E84,0x00004EF2,0x00004F60,0x00004FCE,0x0000503D,0x000050AB,
	0x0000511A,0x00005188,0x000051F7,0x00005266,0x000052D5,0x00005344,0x000053B3,0x00005423,
	0x00005492,0x00005502,0x00005571,0x000055E1,0x00005651,0x000056C1,0x00005731,0x000057A1,
	0x00005812,0x00005882,0x000058F3,0x00005964,0x000059D5,0x00005A46,0x00005AB7,0x00005B28,
	0x00005B99,0x00005C0B,0x00005C7C,0x00005CEE,0x00005D60,0x00005DD2,0x00005E44,0x00005EB6,
	0x00005F28,0x00005F9B,0x0000600E,0x00006080,0x000060F3,0x00006166,0x000061D9,0x0000624D,
	0x000062C0,0x00006334,0x000063A7,0x0000641B,0x0000648F,0x00006503,0x00006577,0x000065EC,
	0x00006660,0x000066D5,0x0000674A,0x000067BF,0x00006834,0x000068A9,0x0000691F,0x00006994,
	0x00006A0A,0x00006A80,0x00006AF6,0x00006B6C,0x00006BE2,0x00006C59,0x00006CCF,0x00006D46,
	0x00006DBD,0x00006E34,0x00006EAB,0x00006F23,0x00006F9A,0x00007012,0x0000708A,0x00007102,
	0x0000717A,0x000071F2,0x0000726B,0x000072E4,0x0000735D,0x000073D6,0x0000744F,0x000074C8,
	0x00007542,0x000075BB,0x00007635,0x000076AF,0x0000772A,0x000077A4,0x0000781F,0x00007899,
	0x00007914,0x0000798F,0x00007A0B,0x00007A86,0x00007B02,0x00007B7E,0x00007BFA,0x00007C76,
	0x00007CF2,0x00007D6F,0x00007DEC,0x00007E68,0x00007EE6,0x00007F63,0x00007FE0,0x0000805E,
	0x000080DC,0x0000815A,0x000081D8,0x00008257,0x000082D6,0x00008355,0x000083D4,0x00008453,
	0x000084D2,0x00008552,0x000085D2,0x00008652,0x000086D2,0x00008753,0x000087D4,0x00008855,
	0x000088D6,0x00008957,0x000089D9,0x00008A5A,0x00008ADC,0x00008B5F,0x00008BE1,0x00008C64,
	0x00008CE7,0x00008D6A,0x00008DED,0x00008E71,0x00008EF4,0x00008F78,0x00008FFD,0x00009081,
	0x00009106,0x0000918B,0x00009210,0x00009295,0x0000931B,0x000093A1,0x00009427,0x000094AD,
	0x00009534,0x000095BA,0x00009641,0x000096C9,0x00009750,0x000097D8,0x00009860,0x000098E8,
	0x00009971,0x000099FA,0x00009A83,0x00009B0C,0x00009B95,0x00009C1F,0x00009CA9,0x00009D34,
	0x00009DBE,0x00009E49,0x00009ED4,0x00009F5F,0x00009FEB,0x0000A077,0x0000A103,0x0000A190,
	0x0000A21C,0x0000A2A9,0x0000A336,0x0000A3C4,0x0000A452,0x0000A4E0,0x0000A56E,0x0000A5FD,
	0x0000A68C,0x0000A71B,0x0000A7AB,0x0000A83A,0x0000A8CA,0x0000A95B,0x0000A9EC,0x0000AA7C,
	0x0000AB0E,0x0000AB9F,0x0000AC31,0x0000ACC3,0x0000AD56,0x0000ADE9,0x0000AE7C,0x0000AF0F,
	0x0000AFA3,0x0000B037,0x0000B0CB,0x0000B160,0x0000B1F5,0x0000B28A,0x0000B320,0x0000B3B5,
	0x0000B44C,0x0000B4E2,0x0000B579,0x0000B610,0x0000B6A8,0x0000B740,0x0000B7D8,0x0000B870,
	0x0000B909,0x0000B9A3,0x0000BA3C,0x0000BAD6,0x0000BB70,0x0000BC0B,0x0000BCA6,0x0000BD41,
	0x0000BDDD,0x0000BE79,0x0000BF15,0x0000BFB2,0x0000C04F,0x0000C0EC,0x0000C18A,0x0000C228,
	0x0000C2C7,0x0000C366,0x0000C405,0x0000C4A5,0x0000C545,0x0000C5E5,0x0000C686,0x0000C727,
	0x0000C7C9,0x0000C86B,0x0000C90D,0x0000C9B0,0x0000CA53,0x0000CAF6,0x0000CB9A,0x0000CC3F,
	0x0000CCE3,0x0000CD89,0x0000CE2E,0x0000CED4,0x0000CF7A,0x0000D021,0x0000D0C8,0x0000D170,
	0x0000D218,0x0000D2C0,0x0000D369,0x0000D413,0x0000D4BC,0x0000D567,0x0000D611,0x0000D6BC,
	0x0000D768,0x0000D814,0x0000D8C0,0x0000D96D,0x0000DA1A,0x0000DAC8,0x0000DB76,0x0000DC25,
	0x0000DCD4,0x0000DD83,0x0000DE33,0x0000DEE4,0x0000DF95,0x0000E046,0x0000E0F8,0x0000E1AB,
	0x0000E25E,0x0000E311,0x0000E3C5,0x0000E479,0x0000E52E,0x0000E5E3,0x0000E699,0x0000E750,
	0x0000E806,0x0000E8BE,0x0000E976,0x0000EA2E,0x0000EAE7,0x0000EBA0,0x0000EC5A,0x0000ED15,
	0x0000EDD0,0x0000EE8B,0x0000EF47,0x0000F004,0x0000F0C1,0x0000F17F,0x0000F23D,0x0000F2FC,
	0x0000F3BB,0x0000F47B,0x0000F53C,0x0000F5FD,0x0000F6BF,0x0000F781,0x0000F844,0x0000F907,
	0x0000F9CB,0x0000FA8F,0x0000FB55,0x0000FC1A,0x0000FCE1,0x0000FDA8,0x0000FE6F,0x0000FF37,
	0x00010000,0x000100C9,0x00010193,0x0001025E,0x00010329,0x000103F5,0x000104C2,0x0001058F,
	0x0001065D,0x0001072B,0x000107FA,0x000108CA,0x0001099A,0x00010A6B,0x00010B3D,0x00010C10,
	0x00010CE3,0x00010DB6,0x00010E8B,0x00010F60,0x00011036,0x0001110C,0x000111E4,0x000112BC,
	0x00011394,0x0001146E,0x00011548,0x00011623,0x000116FE,0x000117DA,0x000118B8,0x00011995,
	0x00011A74,0x00011B53,0x00011C33,0x00011D14,0x00011DF6,0x00011ED8,0x00011FBB,0x0001209F,
	0x00012184,0x00012269,0x00012350,0x00012437,0x0001251F,0x00012607,0x000126F1,0x000127DB,
	0x000128C6,0x000129B2,0x00012A9F,0x00012B8D,0x00012C7C,0x00012D6B,0x00012E5C,0x00012F4D,
	0x0001303F,0x00013132,0x00013226,0x0001331A,0x00013410,0x00013507,0x000135FE,0x000136F7,
	0x000137F0,0x000138EA,0x000139E5,0x00013AE1,0x00013BDF,0x00013CDD,0x00013DDC,0x00013EDC,
	0x00013FDD,0x000140DF,0x000141E2,0x000142E6,0x000143EB,0x000144F0,0x000145F8,0x00014700,
	0x00014809,0x00014913,0x00014A1E,0x00014B2A,0x00014C37,0x00014D46,0x00014E55,0x00014F66,
	0x00015077,0x0001518A,0x0001529E,0x000153B3,0x000154C9,0x000155E0,0x000156F9,0x00015812,
	0x0001592D,0x00015A49,0x00015B66,0x00015C84,0x00015DA4,0x00015EC4,0x00015FE6,0x00016109,
	0x0001622E,0x00016353,0x0001647A,0x000165A2,0x000166CC,0x000167F6,0x00016922,0x00016A4F,
	0x00016B7E,0x00016CAE,0x00016DDF,0x00016F12,0x00017045,0x0001717B,0x000172B1,0x000173E9,
	0x00017523,0x0001765E,0x0001779A,0x000178D7,0x00017A17,0x00017B57,0x00017C99,0x00017DDD,
	0x00017F22,0x00018068,0x000181B0,0x000182F9,0x00018444,0x00018591,0x000186DF,0x0001882F,
	0x00018980,0x00018AD3,0x00018C27,0x00018D7D,0x00018ED5,0x0001902F,0x0001918A,0x000192E6,
	0x00019445,0x000195A5,0x00019707,0x0001986A,0x000199CF,0x00019B36,0x00019C9F,0x00019E0A,
	0x00019F76,0x0001A0E4,0x0001A254,0x0001A3C6,0x0001A53A,0x0001A6B0,0x0001A827,0x0001A9A1,
	0x0001AB1C,0x0001AC9A,0x0001AE19,0x0001AF9A,0x0001B11D,0x0001B2A3,0x0001B42A,0x0001B5B3,
	0x0001B73F,0x0001B8CC,0x0001BA5C,0x0001BBEE,0x0001BD82,0x0001BF18,0x0001C0B0,0x0001C24B,
	0x0001C3E7,0x0001C586,0x0001C727,0x0001C8CB,0x0001CA71,0x0001CC19,0x0001CDC3,0x0001CF70,
	0x0001D11F,0x0001D2D1,0x0001D485,0x0001D63B,0x0001D7F4,0x0001D9B0,0x0001DB6E,0x0001DD2E,
	0x0001DEF1,0x0001E0B7,0x0001E27F,0x0001E44A,0x0001E618,0x0001E7E8,0x0001E9BB,0x0001EB91,
	0x0001ED6A,0x0001EF45,0x0001F123,0x0001F304,0x0001F4E8,0x0001F6CF,0x0001F8B9,0x0001FAA6,
	0x0001FC96,0x0001FE88,0x0002007E,0x00020277,0x00020473,0x00020673,0x00020875,0x00020A7B,
	0x00020C84,0x00020E90,0x0002109F,0x000212B2,0x000214C9,0x000216E2,0x000218FF,0x00021B20,
	0x00021D44,0x00021F6C,0x00022197,0x000223C6,0x000225F9,0x0002282F,0x00022A69,0x00022CA7,
	0x00022EE9,0x0002312F,0x00023378,0x000235C6,0x00023818,0x00023A6D,0x00023CC7,0x00023F25,
	0x00024187,0x000243ED,0x00024658,0x000248C7,0x00024B3A,0x00024DB2,0x0002502E,0x000252AF,
	0x00025534,0x000257BE,0x00025A4D,0x00025CE0,0x00025F78,0x00026215,0x000264B7,0x0002675E,
	0x00026A0A,0x00026CBB,0x00026F71,0x0002722C,0x000274ED,0x000277B3,0x00027A7E,0x00027D4F,
	0x00028026,0x00028302,0x000285E3,0x000288CB,0x00028BB8,0x00028EAB,0x000291A4,0x000294A3,
	0x000297A8,0x00029AB3,0x00029DC4,0x0002A0DC,0x0002A3FB,0x0002A71F,0x0002AA4B,0x0002AD7D,
	0x0002B0B5,0x0002B3F5,0x0002B73B,0x0002BA89,0x0002BDDD,0x0002C139,0x0002C49C,0x0002C807,
	0x0002CB79,0x0002CEF2,0x0002D274,0x0002D5FD,0x0002D98E,0x0002DD27,0x0002E0C8,0x0002E471,
	0x0002E823,0x0002EBDD,0x0002EFA0,0x0002F36C,0x0002F740,0x0002FB1D,0x0002FF04,0x000302F3,
	0x000306EC,0x00030AEF,0x00030EFB,0x00031311,0x00031730,0x00031B5A,0x00031F8E,0x000323CD,
	0x00032816,0x00032C69,0x000330C8,0x00033531,0x000339A6,0x00033E26,0x000342B1,0x00034748,
	0x00034BEB,0x0003509A,0x00035556,0x00035A1D,0x00035EF2,0x000363D3,0x000368C2,0x00036DBD,
	0x000372C6,0x000377DD,0x00037D02,0x00038235,0x00038776,0x00038CC6,0x00039225,0x00039793,
	0x00039D11,0x0003A29E,0x0003A83B,0x0003ADE8,0x0003B3A6,0x0003B974,0x0003BF54,0x0003C545,
	0x0003CB48,0x0003D15C,0x0003D784,0x0003DDBD,0x0003E40A,0x0003EA6A,0x0003F0DE,0x0003F766,
	0x0003FE02,0x000404B4,0x00040B7A,0x00041256,0x00041949,0x00042051,0x00042771,0x00042EA8,
	0x000435F7,0x00043D5E,0x000444DE,0x00044C78,0x0004542B,0x00045BF9,0x000463E1,0x00046BE5,
	0x00047405,0x00047C42,0x0004849B,0x00048D13,0x000495A9,0x00049E5E,0x0004A733,0x0004B029,
	0x0004B940,0x0004C279,0x0004CBD4,0x0004D553,0x0004DEF6,0x0004E8BF,0x0004F2AD,0x0004FCC3,
	0x00050700,0x00051166,0x00051BF5,0x000526B0,0x00053196,0x00053CA8,0x000547E9,0x00055359,
	0x00055EF9,0x00056ACB,0x000576CF,0x00058307,0x00058F75,0x00059C19,0x0005A8F6,0x0005B60C,
	0x0005C35D,0x0005D0EC,0x0005DEB8,0x0005ECC5,0x0005FB14,0x000609A7,0x0006187F,0x0006279F,
	0x00063709,0x000646BF,0x000656C4,0x00066718,0x000677C0,0x000688BD,0x00069A12,0x0006ABC2,
	0x0006BDD0,0x0006D03E,0x0006E310,0x0006F649,0x000709EC,0x00071DFC,0x0007327E,0x00074776,
	0x00075CE6,0x000772D4,0x00078944,0x0007A03A,0x0007B7BB,0x0007CFCD,0x0007E873,0x000801B5,
	0x00081B98,0x00083621,0x00085158,0x00086D43,0x000889E9,0x0008A752,0x0008C586,0x0008E48C,
	0x0009046E,0x00092535,0x000946EB,0x0009699A,0x00098D4D,0x0009B20F,0x0009D7EE,0x0009FEF7,
	0x000A2736,0x000A50BC,0x000A7B97,0x000AA7D9,0x000AD593,0x000B04DA,0x000B35C0,0x000B685C,
	0x000B9CC6,0x000BD317,0x000C0B69,0x000C45DA,0x000C828A,0x000CC199,0x000D032C,0x000D476C,
	0x000D8E82,0x000DD89D,0x000E25EF,0x000E76B0,0x000ECB1B,0x000F2371,0x000F7FFA,0x000FE106,
	0x001046EA,0x0010B206,0x001122C3,0x00119997,0x00121703,0x00129B97,0x001327F6,0x0013BCD5,
	0x00145B00,0x0015035D,0x0015B6F1,0x001676E5,0x0017448D,0x0018216E,0x00190F4A,0x001A102B,
	0x001B2672,0x001C54E8,0x001D9EDA,0x001F0835,0x002095AF,0x00224CFD,0x00243517,0x0026569A,
	0x0028BC49,0x002B73C6,0x002E8E9A,0x003223B0,0x0036519A,0x003B4205,0x00412F4C,0x00486DB7,
	0x00517BB6,0x005D1FF3,0x006CA58F,0x0082608E,0x00A2F8FD,0x00D94C4B,0x0145F2C4,0x028BE5EC,
};

const int squareBuf[] = { 0,1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,
256,289,324,361,400,441,484,529,576,625,676,729,784,841,900,961,1024,1089
,1156,1225,1296,1369,1444,1521,1600,1681,1764,1849,1936,2025,2116,2209,2304
,2401,2500,2601,2704,2809,2916,3025,3136,3249,3364,3481,3600,3721,3844,3969
,4096,4225,4356,4489,4624,4761,4900,5041,5184,5329,5476,5625,5776,5929,6084,
6241,6400,6561,6724,6889,7056,7225,7396,7569,7744,7921,8100,8281,8464,8649,
8836,9025,9216,9409,9604,9801,10000,10201,10404,10609,10816,11025,11236,11449
,11664,11881,12100,12321,12544,12769,12996,13225,13456,13689,13924,14161,14400
,14641,14884,15129,15376,15625,15876,16129,16384,16641,16900,17161,17424,17689
,17956,18225,18496,18769,19044,19321,19600,19881,20164,20449,20736,21025,21316,
21609,21904,22201,22500,22801,23104,23409,23716,24025,24336,24649,24964,25281,
25600,25921,26244,26569,26896,27225,27556,27889,28224,28561,28900,29241,29584,
29929,30276,30625,30976,31329,31684,32041,32400,32761,33124,33489,33856,34225,
34596,34969,35344,35721,36100,36481,36864,37249,37636,38025,38416,38809,39204,
39601,40000,40401,40804,41209,41616,42025,42436,42849,43264,43681,44100,44521,
44944,45369,45796,46225,46656,47089,47524,47961,48400,48841,49284,49729,50176,
50625,51076,51529,51984,52441,52900,53361,53824,54289,54756,55225,55696,56169,
56644,57121,57600,58081,58564,59049,59536,60025,60516,61009,61504,62001,62500,
63001,63504,64009,64516,65025 };
int FastArctan(int x, int y)
{
	int i;
	float z;

	if (x == 0)
	{
		if (y == 0) i = 0;
		else        i = 1023;
	}
	else
	{
		z = ((y < 0) ? (0 - y) : (y)) * 65536u / ((x < 0) ? (0 - x) : (x));
		i = 0;
		if (atan2buf[i + 512] <= z) i += 512;
		if (atan2buf[i + 256] <= z) i += 256;
		if (atan2buf[i + 128] <= z) i += 128;
		if (atan2buf[i + 64] <= z) i += 64;
		if (atan2buf[i + 32] <= z) i += 32;
		if (atan2buf[i + 16] <= z) i += 16;
		if (atan2buf[i + 8] <= z) i += 8;
		if (atan2buf[i + 4] <= z) i += 4;
		if (atan2buf[i + 2] <= z) i += 2;
		if (atan2buf[i + 1] <= z) i += 1;
	}
	i = i * 90 / 1024;

	if (y < 0)
	{
		if (x < 0) return  i - 180;
		else       return -i;
	}
	else
	{
		if (x < 0) return -i + 180;
		else       return  i;
	}
}


const int Rootbuf[] = { 1,1,2,3,4,6,8,11,
16,23,32,45,64,91,128,181,256,362,512,724,1024,1448,2048,2896,4096,5793,8192,11585,16384,23170,32768,46341 };

int FastRoot(int x) {
	if (x<0) {
		cout << "负数无实根！" << endl;
		return 0;
	}
	if (x == 0) { return 0; }
	int i = 0;
	int result;
	for (i = 31; i >0; i--)
	{
		if ((x >> i) & 1)
			break;
	}
	result = Rootbuf[i];
	result = (result + x / result) >> 1;
	result = (result + x / result) >> 1;
	result = (result + x / result) >> 1;
	result = (result + x / result) >> 1;
	result = (result + x / result) >> 1;

	return result;
}





int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	if (NULL == bgrImg.data || NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	int gw = grayImg.cols;
	int gh = grayImg.rows;
	if (gw != width || gh != height)
	{
		cout << "image size is wrong." << endl;
		return MY_FAIL;
	}

	int temp = 0;
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			int b = bgrImg.data[temp++];
			int g = bgrImg.data[temp++];
			int r = bgrImg.data[temp++];

			int grayVal = b * 114 + g * 587 + r * 229;
			//int temp0 = (3 * temp + temp) >> 2;
			grayImg.data[(temp / 3)-1] = grayVal >> 10;
		}
	}
	return MY_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int xwidth = grayImg.cols;
	int xheight = grayImg.rows;
	int ywidth = grayImg.cols;
	int yheight = grayImg.rows;
	if (xwidth != width || xheight != height || ywidth != width || yheight != height)
	{
		cout << "image size is wrong." << endl;
		return MY_FAIL;
	}

	gradImg_x.setTo(0);
	int temp = width + 1;
	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_x =
				grayImg.data[temp - width + 1]
				+ 2 * grayImg.data[temp + 1]
				+ grayImg.data[temp + width + 1]
				- grayImg.data[temp + width - 1]
				- 2 * grayImg.data[temp - 1]
				- grayImg.data[temp + width - 1];

			((float*)gradImg_x.data)[temp + width] = grad_x;
			temp++;
		}
		temp++; temp++;
	}

	gradImg_y.setTo(0);
	temp = width + 1;
	//计算y方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_y =
				grayImg.data[temp + width - 1]
				+ 2 * grayImg.data[temp + width]
				+ grayImg.data[temp + width + 1]
				- grayImg.data[temp - width - 1]
				- 2 * grayImg.data[temp - width]
				- grayImg.data[temp - width + 1];

			((float*)gradImg_y.data)[temp + width] = grad_y;
			temp++;
		}
	}
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat&angleImg, Mat& magImg) {
	if (NULL == gradImg_x.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	int xwidth = gradImg_x.cols;
	int xheight = gradImg_x.rows;
	int ywidth = gradImg_y.cols;
	int yheight = gradImg_y.rows;
	int anglewidth = angleImg.cols;
	int angleheight = angleImg.rows;
	int magwidth = magImg.cols;
	int magheight = magImg.rows;
	if (xwidth != width || xheight != height || ywidth != width || yheight != height || anglewidth != width || angleheight != height || magwidth != width || magheight != height)
	{
		cout << "image size is wrong." << endl;
		return MY_FAIL;
	}
	angleImg.setTo(0);
	magImg.setTo(0);
	int temp = width + 1;
	
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			((float*)angleImg.data)[temp] = FastArctan(((float*)gradImg_x.data)[temp], ((float*)gradImg_y.data)[temp]);
			((float*)magImg.data)[temp] = FastRoot((((float*)gradImg_x.data)[temp]* ((float*)gradImg_x.data)[temp])+ ((float*)gradImg_y.data)[temp]* ((float*)gradImg_y.data)[temp]);

			temp++;
		}
		temp++; temp++;
	}



	return 1;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	if (NULL == grayImg.data || NULL == binaryImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = binaryImg.cols;
	int height = binaryImg.rows;
	int gw = grayImg.cols;
	int gh = grayImg.rows;
	if (gw != width || gh != height)
	{
		cout << "image size is wrong." << endl;
		return MY_FAIL;
	}

	int temp = 0;
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			int pixVal = grayImg.data[temp];
			int dstVal = 0;
			if ((pixVal - th) >> 31)
			{
				dstVal = 0;
			}
			else
			{
				dstVal = 255;
			}
			//binaryImg.at<uchar>(row_i, col_j) = dstVal;
			binaryImg.data[temp++] = dstVal;
		}
	}
	return MY_OK;

}


int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {

	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (hist_len < 256) {
		cout << "not enough space" << endl;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;

	memset(hist, 0, sizeof(hist));
	int temp = 0;
	//计算直方图
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{

			hist[grayImg.data[temp]] += 1;
			temp++;
		}

	}
	return MY_OK;
}


int test_for_difference(Mat grayImg, Mat subImg, int x, int y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int total_diff = 0;
	int temp_for_sub = 0;
	int temp_for_big = y*width + x;
	int width_diff = width - sub_width;
	int temp = 0;
	//遍历模板图上的每一个像素

	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j < sub_width; j++)
		{

			temp = grayImg.data[temp_for_big] - subImg.data[temp_for_sub];
			if (temp >> 31)
				total_diff -= temp;
			else
				total_diff += temp;
			//total_diff = -total_diff;
			//cout << temp << endl;
			temp_for_sub++;
			temp_for_big++;

		}
		temp_for_big += width_diff;
	}
	return total_diff;
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height >= height || sub_width > width) {
		cout << "image size wrong." << endl;
		return MY_FAIL;
	}


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//int temp_for_search = 0;
	int total_diff = 0;
	int i, j, m, n;

	for (i = 0; i < height - sub_height; i++)
	{
		for (j = 0; j < width - sub_width; j++)
		{
			total_diff = test_for_difference(grayImg, subImg, j, i);
			((float*)searchImg.data)[i*width + j] = total_diff;
			total_diff = 0;

		}
	}
	int diffmin = 10000;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			//cout << (int)(((float*)searchImg.data))[i*width + j] ;
			if ((((float*)searchImg.data))[i*width + j] < diffmin)
			{

				diffmin = (((float*)searchImg.data))[i*width + j];
				*x = j; *y = i;
				//cout << diffmin<<endl;
				//cout << "coordinates:" << *x << " " << i << endl;
			}
		}
	}

	//cout << "coordinates:" << *x <<" "<< *y << endl;
	//cout << "coordinates:" << width << " " << height << endl;
	return 1;
}


int test_for_BGRdifference(Mat bgrImg, Mat subImg, int x, int y)
{

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int total_diff = 0;
	int temp_for_sub = 0;
	int temp_for_big = (y*width + x) * 3;
	int width_diff = (width - sub_width) * 3;
	int temp = 0;
	//遍历模板图上的每一个像素
	//cout << temp_for_big << endl;
	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j < sub_width; j++)
		{

			temp = bgrImg.data[temp_for_big] - subImg.data[temp_for_sub];
			if (temp >> 31)
				total_diff -= temp;
			else
				total_diff += temp;
			//total_diff = -total_diff;
			//cout << temp << endl;
			temp_for_sub++;
			temp_for_big++;

			temp = bgrImg.data[temp_for_big] - subImg.data[temp_for_sub];
			if (temp >> 31)
				total_diff -= temp;
			else
				total_diff += temp;
			//total_diff = -total_diff;
			//cout << temp << endl;
			temp_for_sub++;
			temp_for_big++;

			temp = bgrImg.data[temp_for_big] - subImg.data[temp_for_sub];
			if (temp >> 31)
				total_diff -= temp;
			else
				total_diff += temp;
			//total_diff = -total_diff;
			//cout << temp << endl;
			temp_for_sub++;
			temp_for_big++;

		}
		temp_for_big += width_diff;
	}
	return total_diff;
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y){
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height >= height || sub_width > width) {
		cout << "image size wrong." << endl;
		return MY_FAIL;
	}
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//int temp_for_search = 0;
	int total_diff = 0;
	int i = 0, j = 0;
	for (i = 0; i < height - sub_height; i++)
	{
		for (j = 0; j < width - sub_width; j++)
		{
			total_diff = test_for_BGRdifference(colorImg, subImg, j, i);

			//cout << total_diff << endl;
			((float*)searchImg.data)[i*width + j] = total_diff;
			total_diff = 0;

		}
	}
	int diffmin = 100000;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			//cout << (int)(((float*)searchImg.data))[i*width + j] ;
			if ((((float*)searchImg.data))[i*width + j] < diffmin)
			{

				diffmin = (((float*)searchImg.data))[i*width + j];
				*x = j; *y = i;
				//cout << diffmin << endl;
				//cout << "coordinates:" << *x << " " << i << endl;
			}
		}
	}

	//cout << "coordinates:" << *x << " " << *y << endl;
	//cout << "coordinates:" << width << " " << height << endl;
	return 1;
}



float test_for_corrdifference(Mat grayImg, Mat subImg, int x, int y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	float total_diff = 0;
	int temp_for_sub = 0;
	int temp_for_big = y*width + x;
	int width_diff = width - sub_width;
	int temp_add_big = 0;
	int temp_add_sub = 0;
	int temp_add_all = 0;
	//遍历模板图上的每一个像素

	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j < sub_width; j++)
		{

			temp_add_all += grayImg.data[temp_for_big] * subImg.data[temp_for_sub];
			temp_add_big += squareBuf[grayImg.data[temp_for_big]];
			temp_add_sub += squareBuf[subImg.data[temp_for_sub]];
			temp_for_sub++;
			temp_for_big++;

		}
		temp_for_big += width_diff;
	}
	total_diff = (float)temp_add_all / (float)(FastRoot(temp_add_big)*FastRoot(temp_add_sub));
	//cout << total_diff;
	return total_diff;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height >= height || sub_width > width) {
		cout << "image size wrong." << endl;
		return MY_FAIL;
	}
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//int temp_for_search = 0;
	float total_diff = 0;
	int i, j, m, n;
	Mat newcrop = grayImg(Rect(100, 320, 40, 30)).clone();


	for (i = 0; i < height - sub_height; i++)
	{
		for (j = 0; j < width - sub_width; j++)
		{
			total_diff = test_for_corrdifference(grayImg, subImg, j, i);

			((float*)searchImg.data)[i*width + j] = total_diff;
			total_diff = 0;

		}
	}
	float diffmax = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			//cout << (int)(((float*)searchImg.data))[i*width + j] ;
			if ((((float*)searchImg.data))[i*width + j] > diffmax)
			{

				diffmax = (((float*)searchImg.data))[i*width + j];
				*x = j; *y = i;
				//cout << diffmax << endl;
				//cout << "coordinates:" << *x << " " << i << endl;
			}
		}
	}

	//cout << "coordinates:" << *x << " " << *y << endl;
	//cout << "coordinates:" << width << " " << height << endl;
	return 1;
}



int test_for_angledifference(Mat grayImg, Mat subImg, int x, int y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int total_diff = 0;
	int temp_for_sub = sub_width + 1;
	int temp_for_big = (y + 1)*width + x + 1;
	int width_diff = width - sub_width;
	int temp = 0;
	//遍历模板图上的每一个像素

	for (int i = 1; i < sub_height - 1; i++)
	{
		for (int j = 1; j < sub_width - 1; j++)
		{

			temp = ((float*)grayImg.data)[temp_for_big] - ((float*)subImg.data)[temp_for_sub];
			if (temp >> 31)
				total_diff -= temp;
			else
				total_diff += temp;
			if (temp > 180)
				temp = 360 - temp;
			//total_diff = -total_diff;
			//cout << ((float*)grayImg.data)[temp_for_big]<<" "<< ((float*)subImg.data)[temp_for_sub] << endl;
			temp_for_sub++;
			temp_for_big++;

		}
		temp_for_sub++;
		temp_for_big++;
		temp_for_sub++;
		temp_for_big++;
		temp_for_big += width_diff;
	}
	return total_diff;
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int test_x = 15; int test_y = 16;

	if (sub_height >= height || sub_width > width) {
		cout << "image size wrong." << endl;
		return MY_FAIL;
	}
	Mat angleImg(height, width, CV_32FC1);
	angleImg.setTo(0);
	int temp = width + 1;
	//计算梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_x =
				grayImg.data[temp - width + 1]
				+ 2 * grayImg.data[temp + 1]
				+ grayImg.data[temp + width + 1]
				- grayImg.data[temp + width - 1]
				- 2 * grayImg.data[temp - 1]
				- grayImg.data[temp + width - 1];
			int grad_y =
				grayImg.data[temp + width - 1]
				+ 2 * grayImg.data[temp + width]
				+ grayImg.data[temp + width + 1]
				- grayImg.data[temp - width - 1]
				- 2 * grayImg.data[temp - width]
				- grayImg.data[temp - width + 1];

			((float*)angleImg.data)[temp] = FastArctan(grad_x, grad_y);
			temp++;
			//cout << temp;
		}
		temp++; temp++;
	}

	Mat subangleImg(sub_height, sub_width, CV_32FC1);
	subangleImg.setTo(0);
	temp = sub_width + 1;
	//计算x方向梯度图
	for (int row_i = 1; row_i < sub_height - 1; row_i++)
	{
		for (int col_j = 1; col_j < sub_width - 1; col_j += 1)
		{
			int grad_x =
				subImg.data[temp - sub_width + 1]
				+ 2 * subImg.data[temp + 1]
				+ subImg.data[temp + sub_width + 1]
				- subImg.data[temp + sub_width - 1]
				- 2 * subImg.data[temp - 1]
				- subImg.data[temp + sub_width - 1];
			int grad_y =
				subImg.data[temp + sub_width - 1]
				+ 2 * subImg.data[temp + sub_width]
				+ subImg.data[temp + sub_width + 1]
				- subImg.data[temp - sub_width - 1]
				- 2 * subImg.data[temp - sub_width]
				- subImg.data[temp - sub_width + 1];
			((float*)subangleImg.data)[temp] = FastArctan(grad_x, grad_y);

			temp++;
		}
		temp++; temp++;
	}


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//int temp_for_search = 0;
	int total_diff = 0;
	int i, j, m, n;
	//cout << test_for_angledifference(angleImg, subangleImg, 100, 320);
	for (i = 0; i < height - sub_height; i++)
	{
		for (j = 0; j < width - sub_width; j++)
		{
			total_diff = test_for_angledifference(angleImg, subangleImg, j, i);

			//cout << "coord: " << i << " " << j << endl;
			//存储当前像素位置的匹配误差
			//cout << total_diff << endl;
			((float*)searchImg.data)[i*width + j] = total_diff;
			total_diff = 0;

		}
	}
	int diffmin = 100000;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			//cout << (int)(((float*)searchImg.data))[i*width + j] ;
			if ((((float*)searchImg.data))[i*width + j] < diffmin)
			{

				diffmin = (((float*)searchImg.data))[i*width + j];
				*x = j; *y = i;
				//cout << diffmin << endl;
				//cout << "coordinates:" << *x << " " << i << endl;
			}
		}
	}

	//cout << "coordinates:" << *x <<" "<< *y << endl;
	//cout << "coordinates:" << width << " " << height << endl;
	return 1;
}


int test_for_lengthdifference(Mat grayImg, Mat subImg, int x, int y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int total_diff = 0;
	int temp_for_sub = sub_width + 1;
	int temp_for_big = (y + 1)*width + x + 1;
	int width_diff = width - sub_width;
	int temp = 0;
	//遍历模板图上的每一个像素

	for (int i = 1; i < sub_height - 1; i++)
	{
		for (int j = 1; j < sub_width - 1; j++)
		{

			temp = ((float*)grayImg.data)[temp_for_big] - ((float*)subImg.data)[temp_for_sub];
			if (temp >> 31)
				total_diff -= temp;
			else
				total_diff += temp;
			//total_diff = -total_diff;
			//cout << ((float*)grayImg.data)[temp_for_big]<<" "<< ((float*)subImg.data)[temp_for_sub] << endl;
			temp_for_sub++;
			temp_for_big++;

		}
		temp_for_sub++;
		temp_for_big++;
		temp_for_sub++;
		temp_for_big++;
		temp_for_big += width_diff;
	}
	return total_diff;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int test_x = 15; int test_y = 16;

	if (sub_height >= height || sub_width > width) {
		cout << "image size wrong." << endl;
		return MY_FAIL;
	}
	Mat angleImg(height, width, CV_32FC1);
	angleImg.setTo(0);
	int temp = width + 1;
	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_x =
				grayImg.data[temp - width + 1]
				+ 2 * grayImg.data[temp + 1]
				+ grayImg.data[temp + width + 1]
				- grayImg.data[temp + width - 1]
				- 2 * grayImg.data[temp - 1]
				- grayImg.data[temp + width - 1];
			int grad_y =
				grayImg.data[temp + width - 1]
				+ 2 * grayImg.data[temp + width]
				+ grayImg.data[temp + width + 1]
				- grayImg.data[temp - width - 1]
				- 2 * grayImg.data[temp - width]
				- grayImg.data[temp - width + 1];
			/*if (row_i == (320+test_y) && col_j == (100+test_x))
			cout << grad_x << " and " << grad_y << " and " << FastArctan(grad_x, grad_y) << " and " << (test_y + 320) * width + (100 + test_x) << " and " << temp << endl;
			//((float*)gradImg_y.data)[temp + width] = grad_y;*/
			((float*)angleImg.data)[temp] = FastRoot(grad_x*grad_x + grad_y* grad_y);
			temp++;
			//cout << temp;
		}
		temp++; temp++;
	}

	Mat subangleImg(sub_height, sub_width, CV_32FC1);
	subangleImg.setTo(0);
	temp = sub_width + 1;
	//计算x方向梯度图
	for (int row_i = 1; row_i < sub_height - 1; row_i++)
	{
		for (int col_j = 1; col_j < sub_width - 1; col_j += 1)
		{
			int grad_x =
				subImg.data[temp - sub_width + 1]
				+ 2 * subImg.data[temp + 1]
				+ subImg.data[temp + sub_width + 1]
				- subImg.data[temp + sub_width - 1]
				- 2 * subImg.data[temp - 1]
				- subImg.data[temp + sub_width - 1];
			int grad_y =
				subImg.data[temp + sub_width - 1]
				+ 2 * subImg.data[temp + sub_width]
				+ subImg.data[temp + sub_width + 1]
				- subImg.data[temp - sub_width - 1]
				- 2 * subImg.data[temp - sub_width]
				- subImg.data[temp - sub_width + 1];
			((float*)subangleImg.data)[temp] = FastRoot(grad_x*grad_x + grad_y* grad_y);

			temp++;
		}
		temp++; temp++;
	}

	//cout << ((float*)subangleImg.data)[test_y * sub_width + test_x] << " and " << ((float*)angleImg.data)[(test_y+320) * width +(100+test_x)] << endl;


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//int temp_for_search = 0;
	int total_diff = 0;
	int i, j, m, n;
	//cout << test_for_angledifference(angleImg, subangleImg, 100, 320);
	for (i = 0; i < height - sub_height; i++)
	{
		for (j = 0; j < width - sub_width; j++)
		{
			total_diff = test_for_angledifference(angleImg, subangleImg, j, i);

			//cout << "coord: " << i << " " << j << endl;
			//存储当前像素位置的匹配误差
			//cout << total_diff << endl;
			((float*)searchImg.data)[i*width + j] = total_diff;
			total_diff = 0;

		}
	}
	int diffmin = 100000;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			//cout << (int)(((float*)searchImg.data))[i*width + j] ;
			if ((((float*)searchImg.data))[i*width + j] < diffmin)
			{

				diffmin = (((float*)searchImg.data))[i*width + j];
				*x = j; *y = i;
				//cout << diffmin << endl;
				//cout << "coordinates:" << *x << " " << i << endl;
			}
		}
	}

	//cout << "coordinates:" << *x <<" "<< *y << endl;
	//cout << "coordinates:" << width << " " << height << endl;
	return 1;
}


int test_for_histdifference(Mat grayImg, int*hist, int x, int y, int range, int sub_width, int sub_height)
{
	int a = x; int b = y;
	Mat ROI = grayImg(Rect(a, b, sub_width, sub_height)).clone();
	int ROIhist[256]; memset(ROIhist, 0, sizeof(ROIhist));
	int temp;
	int flag = ustc_CalcHist(ROI, ROIhist, 256);
	//printhist(ROIhist);
	int total_diff = 0;
	for (int i = 0; i < range; i++) {
		temp = hist[i] - ROIhist[i];
		if (temp >> 31)
			total_diff -= temp;
		else
			total_diff += temp;
	}
	//cout << y << " " << x << " " << total_diff<<"inside" << endl;
	//cout << total_diff;
	return total_diff;
}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int histBuff[256];
	memset(histBuff, 0, sizeof(histBuff));
	//printhist(histBuff);
	int flag = ustc_CalcHist(subImg, histBuff, 256);

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height >= height || sub_width > width) {
		cout << "image size wrong." << endl;
		return MY_FAIL;
	}
	//int test_x = 15; int test_y = 16;

	//cout << test_for_histdifference(grayImg, histBuff,100, 320, 256, sub_width, sub_height) << endl;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//int temp_for_search = 0;
	int total_diff = 0;
	int i, j, m, n;
	
	for (i = 0; i < height - sub_height; i++)
	{
		for (j = 0; j < width - sub_width; j++)
		{
			total_diff = test_for_histdifference(grayImg, histBuff, j, i, 256, sub_width, sub_height);

			//cout << "coord: " << i << " " << j << endl;
			//存储当前像素位置的匹配误差
			//cout << i<<" "<<j<<" "<<total_diff << endl;
			((float*)searchImg.data)[i*width + j] = total_diff;
			total_diff = 0;

		}
	}
	int diffmin = 100000;
	for (int i = 100; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			//cout << (int)(((float*)searchImg.data))[i*width + j] ;
			if ((((float*)searchImg.data))[i*width + j] < diffmin)
			{

				diffmin = (((float*)searchImg.data))[i*width + j];
				*x = j; *y = i;
				//cout << diffmin << endl;
				//cout << "coordinates:" << *x << " " << i << endl;
			}
		}
	}

	//cout << "coordinates:" << *x << " " << *y << endl;
	//cout << "coordinates:" << width << " " << height << endl;
	return 1;
}


