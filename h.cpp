/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

extern int CARDPRINTMODE;

enum Suit
{
  S_CLUBS,
  S_DIAMONDS,
  S_HEARTS,
  S_SPADES,
  S_UNKNOWN
};

class Card
{
  /*
  CARDS

  There are 13 values. They are as follows when represented by an integer and by a character:
  2=2, 3=3, 4=4, 5=5, 6=6, 7=7, 8=8, 9=9, 10=T, 11=J, 12=Q, 13=K, 14=A
  */

  public:
    Suit suit;
    int value; //2-14 (ace is not represented by 1 but by 14, because it's the highest for Poker)

  public:

    Card(int value, Suit suit);
    Card(int index);
    Card(); //makes invalid card
    Card(const std::string& shortName); //e.g. "Qs"
    Card(const Card& other);

    int getValue() const; //returns 2 for 2, up to 14 for ace
    void setValue(int value); //value must be 2-14, for ace use 14
    Suit getSuit() const;
    void setSuit(Suit suit);

    /*
    getIndex gives just some of the many possible integer representations of cards. This index
    could be considered the standard OOPoker index for cards. But the fast hand evaluations
    algorithms from pokereval.h all use their own, different, index systems.
    OOPoker Index Values:
    unknown : -1
    clubs   :  0=ace,  1=2,  2=3,  3=4,  4=5,  5=6,  6=7,  7=8,  8=9,  9=T, 10=J, 11=Q, 12=K
    diamonds: 13=ace, 14=2, 15=3, 16=4, 17=5, 18=6, 19=7, 20=8, 21=9, 22=T, 23=J, 24=Q, 25=K
    hearts  : 26=ace, 27=2, 28=3, 29=4, 30=5, 31=6, 32=7, 33=8, 34=9, 35=T, 36=J, 37=Q, 38=K
    spades  : 39=ace, 40=2, 41=3, 42=4, 43=5, 44=6, 45=7, 46=8, 47=9, 48=T, 49=J, 50=Q, 51=K
    */
    int getIndex() const;
    void setIndex(int index);

    std::string getShortName() const; //returns e.g. "Ah" for ace of hearts
    std::string getShortNameAscii() const; //returns short name using ascii card symbols for the suit
    std::string getShortNameUnicode() const; //returns short name using unicode card symbols for the suit
    std::string getShortNamePrintable() const; //returns short name using either ascii or unicode symbols for the suit, depending on what works for your operating system
    std::string getLongName() const; //returns something like "Ace of Hearts"

    //set the card from a name such as "Ah" for ace of hearts
    void setShortName(const std::string& name);

    //if this returns false, the card is unknown
    //some functions return an invalid card to indicate "no combination" or so
    bool isValid() const;
    void setInvalid(); //sets this card to invalid (unknown)

};

char valueToSymbol(int value);

int compare(const Card& a, const Card& b);
bool cardGreater(const Card& a, const Card& b);

// string of short names to vector of card indices
// e.g. string Ah5s becomes vector {26,43}
std::vector<int> cardNamesToIndices(const std::string& names);
/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "card.h"

enum Command
{
  A_FOLD,
  A_CHECK,
  A_CALL,
  A_RAISE //also used to BET. Requires amount to be given, and amount must be amount of chips moved to table, not the amount raises with.
};

struct Action
{
  Command command;
  int amount; //Only used for the A_RAISE command. This is NOT the raise amount. This is the total value of money you move from your stack to the pot. So if the call amount was 50, and you raise with 100, then this amount must be set to 150, not 100.

  Action(Command command, int amount = 0);
  Action();
};

/*
Is the action allowed by the game of no-limit Texas Hold'em?
It is not allowed if:
-you need to move more chips to the table than you have in your stack to perform this action (unless you go all-in)
-it's a raise action but the amount of chips raised is smaller than the highest raise so far this deal (unless you go all-in)
-it's a check action while the call amount is higher than 0

action: the action to test
stack: stack the player currently has (excludes his wager)
wager: chips the player has contributed to the pot this deal so far
highestWager: the amount of wager of the player with highest wager (needed to know amount required to call)
highestRaise: the highest raise amount so far during this deal (the amount raised above the call amount) (needed to exclude small raises)
*/
bool isValidAction(const Action& action, int stack, int wager, int highestWager, int highestRaise);

/*
Is this an all-in action, and, is it valid?
It is not consudered valid if it's a raise action and the amount of chips is larger than your stack. It
must be exactly equal to be an all-in action.
*/
bool isValidAllInAction(const Action& action, int stack, int wager, int highestWager, int highestRaise);
/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "action.h"

struct Event;
struct Info;

class AI //interface class, used for bots, but also for human players (then the AI uses human input instead of calculating itself)
{
  public:

    virtual ~AI(){}

    /*
    doTurn:
    make a decision for this turn: fold, check, call or raise?
    */
    virtual Action doTurn(const Info& info) = 0;

    /*
    onEvent:
    process events if needed (not required, can be used for extra information)
    */
    virtual void onEvent(const Event& event);

    /*
    boastCards:
    called at the end of a deal, only if this AI wasn't required to show his cards.
    */
    virtual bool boastCards(const Info& info);

    /*
    wantsToLeave:
    sometimes you get a choice to leave the table. NOTE: in OOPoker this is allowed only for human players. So
    this function will never be called on non-human AI's, but it's still possible to implement it.
    Return true if you want to leave the table, false otherwise.
    */
    virtual bool wantsToLeave(const Info& info);

    /*
    getAIName:
    This is not the name of the player, but the name of his "type of brains".
    */
    virtual std::string getAIName() = 0;
};

/*
OOPoker Changelist

20100513:

-added combinatorial mathematics functions in pokermath.h: such as factorial and binomial coefficient (combination)
-added two console utilities to calculate poker odds (with human interface rather than meant for AI's)


20100512:

Incompatible interface changes:

-renamed "BettingStructure" to "Rules" (and variable names to rules)
-renamed "holdCards" from Info struct, it's getHoleCards() now, because this info is already in the player struct in the info struct.
-renamed "lastRaiseAmount" to "minRaiseAmount" in Info struct (to support other rules about minimum raise amount)

Other changes:

-made the game-logic at least 10 times faster (when using AICall bots) by not copying structs and std::vectors all the time anymore
-found a faster hand evaluator for 7 cards, which also doesn't need a 124MB cache file anymore. This makes OOPoker a lot more user friendly and faster.

20100510:

Incompatible interface changes:

-changed cards in the Info struct into std::vectors
-changed interface of Host class to use Info instead of Table as parameter
-changed the variable name "tableCard" into "boardCard" everywhere.
-changed the variable name "handCard" into "holeCard" everywhere.
-changed struct name "PlayerTurnInfo" into "PlayerInfo"

Other changes:

-fixed bug that could cause infinite loop in settling bets and huge log file (if
 someone went all-in with small amount, the rest checked, and all-in player had
 last raise index due to that)
-added eval7 benchmark in the unit test
-shortened readme.txt a bit.
-changed the formula of "Went to showdown" statistic: now only showdown percentage
 when you already reached the flop are shown (so it's now showdowns_seen / flops_seen
 instead of showdowns_seen / deals)
-added more comma's in player statistics log output
-made message for human player terminal input a bit shorter

*/
/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

#include "card.h"

/*
Combination:
Represents a combination of 5 cards, forming things like a pair, a flush, etc..
This file also has functions for checking which combination is formed by 5 or by 7 cards.

This contains a naive and slow 7-card hand evaluator, but has the advantage that
these functions can be used to check what combination a player has and to convert it to a name,
and to check which 5 cards are involved.

If you need to evaluate many of hands for combinations for a poker AI, then don't use
the functions from this file. Use eval7 from pokermath.h instead: that one can evaluate
millions of hands per second.
*/


enum ComboType
{
  C_HIGH_CARD,
  C_PAIR,
  C_TWO_PAIR,
  C_THREE_OF_A_KIND,
  C_STRAIGHT,
  C_FLUSH,
  C_FULL_HOUSE,
  C_FOUR_OF_A_KIND,
  C_STRAIGHT_FLUSH //includes royal flush
};

struct Combination
{
  ComboType type;

  /*
  the cards are the 5 cards involved in this combo, and they are sorted in the following order (depending on the type of combo):
  Straight Flush: highest to lowest (Royal Flush: ace to 10)
  Four of a kind: first the 4 cards of the same value, then the 5th card (color order unspecified)
  Full House: first the 3 cards, then the 2 cards (color order unspecified)
  Flush: highest to lowest
  Straight: highest to lowest
  Three of a kind: first the 3, then the highest other cards, then the lowest other card
  Two Pair: first the 2 of the highest pair, then the two of the lowest pair, then the extra card
  Pair: first the 2, then the other 3 sorted from highest to lowest
  High card: highest to lowest
  */
  Card cards[5];
  int cards_used; //this is normally 5, unless the combo was made out of less cards (e.g. three of a kind detected given 3 cards)

  std::string getName() const;
  std::string getNameWithAllCards() const;
  std::string getNameWithAllCardsPrintable() const;
};

////////////////////////////////////////////////////////////////////////////////

//only call after having checked all combinations
bool checkHighCard(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations (including two pairs, three of a kind, or other things that already contain a pair in them too)
bool checkPair(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkTwoPair(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations (including full house, four of a kind, or other things that already contain a pair in them too)
bool checkThreeOfAKind(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkStraight(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkFlush(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkFullHouse(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkFourOfAKind(Card result[5], const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkStraightFlush(Card result[5], const std::vector<Card>& sorted);

////////////////////////////////////////////////////////////////////////////////

//only call after already having checked for better combinations (including two pairs, three of a kind, or other things that already contain a pair in them too)
bool checkPair(const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkTwoPair(const std::vector<Card>& sorted);

//only call after already having checked for better combinations (including full house, four of a kind, or other things that already contain a pair in them too)
bool checkThreeOfAKind(const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkStraight(const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkFlush(const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkFullHouse(const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkFourOfAKind(const std::vector<Card>& sorted);

//only call after already having checked for better combinations
bool checkStraightFlush(const std::vector<Card>& sorted);

////////////////////////////////////////////////////////////////////////////////

void sortCardsHighToLow(std::vector<Card>& cards);

////////////////////////////////////////////////////////////////////////////////

/*
Returns the best combo formed by the given cards. It doesn't matter how many
cards are given, but it must be at least 5.
*/
void getCombo(Combination& combo, const std::vector<Card>& cards);
void getCombo(Combination& combo, const Card& card1, const Card& card2, const Card& card3, const Card& card4, const Card& card5);

//2 cards given by short names
void getCombo(Combination& combo
            , const std::string& card1
            , const std::string& card2);
//3 cards given by short names
void getCombo(Combination& combo
            , const std::string& card1
            , const std::string& card2
            , const std::string& card3);
//4 cards given by short names
void getCombo(Combination& combo
            , const std::string& card1
            , const std::string& card2
            , const std::string& card3
            , const std::string& card4);
//5 cards given by short names
void getCombo(Combination& combo
            , const std::string& card1
            , const std::string& card2
            , const std::string& card3
            , const std::string& card4
            , const std::string& card5);
//6 cards given by short names
void getCombo(Combination& combo
            , const std::string& card1
            , const std::string& card2
            , const std::string& card3
            , const std::string& card4
            , const std::string& card5
            , const std::string& card6);
//7 cards given by short names
void getCombo(Combination& combo
            , const std::string& card1
            , const std::string& card2
            , const std::string& card3
            , const std::string& card4
            , const std::string& card5
            , const std::string& card6
            , const std::string& card7);
//string.size() / 2 cards given by short names combined in one string
void getCombo(Combination& combo, const std::string& cards);


//returns -1 if combo a is worth less than combo b, 0 if worth the same, 1 if combo a is worth more than combo b.
int compare(const Combination& a, const Combination& b);

bool combinationGreater(const Combination& a, const Combination& b);


/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "card.h"


class Deck
{
  /*
  Deck of cards, that can be randomly shuffled using the true random.h, and
  allows easily selecting next cards.
  */

  private:

    Card cards[52]; //card 0 is the top card
    int index;

  public:

    Deck();
    void shuffle();
    Card next(); //never call this more than 52 times in a row.
};
/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>

#include "card.h"

struct Player;
class Observer;

enum EventType
{
  //an EventType has some information associated with it in the Event struct.
  //The comment at to each event says which info exactly, if any.
  //If an event is related to a player, the player is always given as a string (not as an index). You can use those player names to uniquely identify them.

  //info used: player, chips (with how much chips this player joins)
  E_JOIN, //player joins table

  //info used: player, chips
  E_QUIT, //player quits table (with so many chips left; usually 0 unless he quits early, which isn't always allowed)

  //info used: player, chips (with how much chips this player rebuys)
  E_REBUY, //player lost his stack and rebuys

  //info used: player, chips
  E_SMALL_BLIND, //player places the small blind on the table. The amount can be smaller than the actual small blind, because, he might be all-in!

  //info used: player, chips
  E_BIG_BLIND, //player places the small blind on the table. The amount can be smaller than the actual big blind, because, he might be all-in!

  //info used: player, chips
  E_ANTE, //player places the ante on the table. The amount can be smaller than the actual ante, because, he might be all-in!

  //info used: player
  E_FOLD,

  //info used: player
  E_CHECK,

  //info used: player
  E_CALL,

  //info used: player, chips (the amount ABOVE the call amount)
  E_RAISE,

  //info used: smallBlind, bigBlind, ante
  E_NEW_DEAL, //= going back to preflop, receiving cards, ...

  //info used: player, card1, card2 (the hand cards)
  E_RECEIVE_CARDS, //This event shows the holecards you get.

  //info used: card1, card2, card3
  E_FLOP, //this event contains the 3 flop cards

  //info used: card1, card2, card3, card4 (the turn card)
  E_TURN, //this event contains the Turn card

  //info used: card1, card2, card3, card4, card5 (the river card)
  E_RIVER, //this event contains the River card

  //info used: none
  E_SHOWDOWN, //this event indicates that the stage past the river is reached while multiple players are still active. A showdown of their cards will follow. This event can be used to distinguish between a deal ending because one player outbluffed everyone, or, a showdown going to occur. Not to be confused with E_PLAYER_SHOWDOWN!

  //info used: chips (total pot)
  E_POT_DIVISION, //this event gives the pot amount right before a win amount, so you know the difference between split pot and non-split-pot. This even also indicates the deal is done, it is always given exactly once per deal (and is a counterpart of E_NEW_DEAL).

  //info used: player, card1, card2 (his hand cards)
  E_PLAYER_SHOWDOWN, //the cards shown by 1 player, when required to show them. Not to be confused with E_SHOWDOWN!

  //info used: player, card1, card2 (his hand cards)
  E_BOAST, //this is when the player shows cards while not needed, for the rest same as E_PLAYER_SHOWDOWN

  //info used: player, card1, card2, card3, card4, card
  E_COMBINATION, //combination a player has after his showdown

  //info used: player, chips (amount that goes from the pot towards this player)
  E_WIN, //player wins the entire pot or part of the pot at the end of a deal

  //info used: player
  E_DEALER, //lets know who the dealer is

  //info used: player, position, chips (chips is used to indicate tournament score, can be stack minus buyInTotal for example, depending on win condition)
  E_TOURNAMENT_RANK, //how good did this player rank for this tournament?

  //info used: player, ai
  E_REVEAL_AI, //reveals the AI of a player (at the end of the game)

  //info used: message
  E_LOG_MESSAGE, //not sent to AI's

  //info used: message
  E_DEBUG_MESSAGE, //not sent to AI's

  E_NUM_EVENTS //don't use
};

struct Event
{
  EventType type;

  std::string player; //name of player the event is related to
  std::string ai; //used for very rare events that unmistify the AI of a player
  int chips; //money above call amount, if it's a raise event. Win amount if it's a win event. Pot amount if it's a pot event.

  int smallBlind;
  int bigBlind;
  int ante;

  int position; //position for E_TOURNAMENT_WIN event

  //cards used for some event. Flop uses 3, turn uses card4, river uses card5, showdown and new_game uses card1 and card2. Win uses all 5.
  Card card1;
  Card card2;
  Card card3;
  Card card4;
  Card card5;

  Event(EventType type);
  Event(EventType type, const std::string& player);
  Event(EventType type, const std::string& player, int chips);
  Event(EventType type, int position, const std::string& player);
  Event(EventType type, int position, int chips, const std::string& player);
  Event(EventType type, const std::string& player, const std::string& ai);
  Event(EventType type, const Card& card1);
  Event(EventType type, const Card& card1, const Card& card2);
  Event(EventType type, const Card& card1, const Card& card2, const Card& card3);
  Event(EventType type, const Card& card1, const Card& card2, const Card& card3, const Card& card4);
  Event(EventType type, const Card& card1, const Card& card2, const Card& card3, const Card& card4, const Card& card5);
  Event(EventType type, const std::string& player, const Card& card1, const Card& card2, const Card& card3, const Card& card4, const Card& card5);
  Event(EventType type, const std::string& player, const Card& card1, const Card& card2);
  Event(EventType type, int smallBlind, int bigBlind, int ante);
  Event(const std::string& message, EventType type);


  std::string message;
};

//this gives the event in a good form for a log or computer parsing
std::string eventToString(const Event& event);

//this gives the event in a more verbose full English sentence form
std::string eventToStringVerbose(const Event& event);

//TODO: make the opposite, a stringToEvent parsing function

//sends unprocessed events to player, but only events the player is allowed to know! (the events vector is not supposed to contain personal events, such as E_RECEIVE_CARDS)
void sendEventsToPlayers(size_t& counter, std::vector<Player>& players, std::vector<Observer*>& observers, const std::vector<Event>& events);



/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>

#include "info.h"


//forward declarations
class Host;
class Table;
struct Player;
class Observer;
struct Event;

void makeInfo(Info& info, const Table& table, const Rules& rules, int playerViewPoint);

void dividePot(std::vector<int>& wins, const std::vector<int>& bet, const std::vector<int>& score, const std::vector<bool>& folded);
int getNumActivePlayers(const std::vector<Player>& players);
bool betsSettled(int lastRaiseIndex, int current, int prev_current, const std::vector<Player>& players);

class Game
{
  private:
    Host* host;
    std::vector<Player> playersOut; //players who quit the table (remembered in order to give the win rankings at the end). Not used if rebuys are allowed, since players then stay.

    std::vector<Player> players;
    std::vector<Observer*> observers;
    std::vector<Event> events;

    size_t eventCounter;
    int numDeals; //how much deals are done since the game started

    Rules rules;
    
    Info infoForPlayers; //this is to speed up the game a lot, by not recreating the Info object everytime

  protected:
    void settleBets(Table& table, Rules& rules);
    void kickOutPlayers(Table& table);
    void declareWinners(Table& table);
    void sendEvents(Table& table);
    const Info& getInfoForPlayers(Table& table, int viewPoint = -1); //rather heavy-weight function! Copies entire Info object.

  public:

    Game(Host* host); //The game class will NOT delete the host, you have to clean up this variable yourself if needed.
    ~Game();

    //The Game class will take care of deleting the AI's and observers in its desctructor.
    void addPlayer(const Player& player);
    void addObserver(Observer* observer);
    void setRules(const Rules& rules);

    void runTable(Table& table);

    void doGame();
};

/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

struct Info;

/*
The Host is someone who runs a Game. The Host can stop the game, and receive events from the game.
*/
class Host
{
  public:
    virtual ~Host(){}
   
    virtual void onFrame() = 0; //called between every player decision
    virtual void onGameBegin(const Info& info) = 0; //called after all players are sitting at the table, right before the first deal starts
    virtual void onDealDone(const Info& info) = 0;
    virtual void onGameDone(const Info& info) = 0; //when the whole tournament is done

    virtual bool wantToQuit() const = 0;
    virtual void resetWantToQuit() = 0;
};

/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "host.h"

/*
Implementation of Host that uses the terminal.
*/
class HostTerminal : public Host
{
  private:
    bool quit;

    bool human_detected; //is used to print messages in certain way if the human player is out.
    
    int dealCount;

  public:

    HostTerminal();

    virtual void onFrame(); //called between every player decision
    virtual void onGameBegin(const Info& info); //called after all players are sitting at the table, right before the first deal starts
    virtual void onDealDone(const Info& info);
    virtual void onGameDone(const Info& info); //when the whole tournament is done

    virtual bool wantToQuit() const;
    virtual void resetWantToQuit();

    //not part of the Host interface, additial communication for the terminal-based Human AI and/or Observer
    void setQuitSignalFromHumanPlayer(); //command given by AIHuman to HostTerminal
    void setHasHumanPlayer(bool has);
};

/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "rules.h"
#include "event.h"

struct Action;

/*
Info has the same information as Table, but only the information that is visible
for a certain player, and presented such that modifying it doesn't cause harm to
the game itself, since it's only a copy of the information.

It also contains various utility methods.

This is the Info an AI gets for each decision.

pokermath.h
card.h
combination.h
random.h

All other header files are probably less useful for an AI (they're for running the game)
*/

#include "action.h"
#include "rules.h"

//info about a player for a turn during the betting
struct PlayerInfo
{
  bool folded; //if true, this player has already folded for this game. Either just now (if his action has FOLD in it), or earlier (if his action has ACTION_NONE in it).

  std::string name; //name of the player
  int stack;
  int wager; //how much money this player has bet during the whole game so far (where game is one hand)

  Action lastAction; //what the player did this turn (most recent action of this player)

  bool showdown; //if true, the hand card values of this player are stored in the holeCard variables.
  std::vector<Card> holeCards;

  PlayerInfo();

  const std::string& getName() const;

  bool isAllIn() const;
  bool isOut() const; //can't play anymore, has no more money
  bool isFolded() const;
  bool canDecide() const; //returns true if stack > 0 and not folded
};

struct Info //all the info a player gets during a decision
{
  ///Personal Info

  int yourIndex; //your index in the players vector of the table, or -1 if this info is global. This is not your position. Use getPosition to get your position compared to the dealer.

  ///Global Info

  int dealer; //index of the dealer
  int current; //index of the player currently making a decision

  Round round;
  int turn; //the number of times you had to decide for this Round (where Round is pre-flop, flop, etc...). Normally this is 1. If someone raised causing the betting to go round again, this increases. So this is kind of a "sub-round" index in fact

  int minRaiseAmount; //minimum raise amount *above the call amount* to be able to raise according to the game rules

  //NOTE: the values of these cards are only valid if the Round is correct.
  std::vector<Card> boardCards; //the community cards on the table. 0 pre-flop, 3 at flop, 4 at turn, 5 at river.

  std::vector<PlayerInfo> players; //you yourself are included in this vector, at yourIndex

  Rules rules;

  ///Constructor

  Info();

  bool isGlobal() const; //if this returns true, then this Info is NOT about you as player, but contains only the globally known information. If this returns true, do NOT use any of the functions that involve you (such as getMRatio(), getSmallBlind(), ...)

  ///Personal Utility methods. Only use if isGlobal() returns false.

  const PlayerInfo& getYou() const;

  const std::vector<Card>& getHoleCards() const; //shortcut to your hole cards

  int getCallAmount() const; //get amount of money required for you to call
  int getMinChipsToRaise() const; //get amount of chips you need to move to the table to raise with the minimum raise amount. This is getCallAmount() + lastRaiseAmount

  int getPosition() const; //returns 0 if you're dealer, 1 for sb, 2 for bb, 3 = under the gun, getNumPlayers()-1 = cut-off.

  int getStack() const; //get your stack
  int getWager() const; //get your wager

  //For more statistics like this, see pokermath.h
  double getMRatio() const; //returns (your stack) / (small blind + big blind + total antes), in other words, the number of laps you can still survive with your current stack
  double getPotOdds() const; //this gives getPot() / getCallAmount(). Can be infinite if callamount is 0. Higher is better.
  double getPotOddsPercentage() const; //gets pot odds as a percengate. Gives callAmount / (total pot + callAmount). For example if the pot odds are 2:1, then the percentage is 33.3% (and the return value is 0.33 since it's a number in the range 0.0-1.0)
  double getPotEquity() const; //see description in pokermath.h for more information about this function. This here is just a convenience wrapper.

  //get std::vectors of cards, handy for calling some of the mathematical functions
  std::vector<Card> getHandTableVector() const;

  /*
  Is the action allowed by the game?
  It is not allowed if:
  -you need to move more chips to the table than you have in your stack to perform this action (unless you go all-in)
  -it's a raise action but the amount of chips raised is smaller than the highest raise so far this deal (unless you go all-in)
  -it's a check action while the call amount is higher than 0
  */
  bool isValidAction(const Action& action) const;
  bool isValidAllInAction(const Action& action) const; //will this action bring you all-in? This function must be called before the action is performed (after it's performed your stack is 0 and then the checks in this function don't work anymore)

  Action getCheckFoldAction() const; //checks if possible, folds otherwise
  Action amountToAction(int amount) const; //converts a number (representing stack amount you offer), to an action. If the number is greater than your stack, it'll make it an all-in action the size of your stack instead. The returned action will be a valid action, no matter what amount given. If the amount has to be changed, it'll always be smaller, not bigger, than the given amount.
  Action getCallAction() const; //returns call action if call amount > 0, check action otherwise
  Action getRaiseAction(int raise) const; //calls amountToAction with getCallAmount() added
  Action getAllInAction() const; //creates action of type call or raise depending on  your stack size and call amount, so that you're all-in


  ///Global versions of the per-player utility methods. Allows giving player index.

  const std::vector<Card>& getHoleCards(int index) const; //shortcut to your hole cards

  int getCallAmount(int index) const; //get amount of money required for you to call
  int getMinChipsToRaise(int index) const; //get amount of chips you need to move to the table to raise with the minimum raise amount. This is getCallAmount() + lastRaiseAmount

  int getPosition(int index) const; //returns 0 if you're dealer, 1 for sb, 2 for bb, 3 = under the gun, getNumPlayers()-1 = cut-off.

  int getStack(int index) const; //get your stack
  int getWager(int index) const; //get your wager

  //For more statistics like this, see pokermath.h
  double getMRatio(int index) const; //returns (your stack) / (small blind + big blind + total antes), in other words, the number of laps you can still survive with your current stack
  double getPotOdds(int index) const; //this gives getPot() / getCallAmount(). Can be infinite if callamount is 0. Higher is better.
  double getPotOddsPercentage(int index) const; //gets pot odds as a percengate. Gives callAmount / (total pot + callAmount). For example if the pot odds are 2:1, then the percentage is 33.3% (and the return value is 0.33 since it's a number in the range 0.0-1.0)
  double getPotEquity(int index) const; //see description in pokermath.h for more information about this function. This here is just a convenience wrapper.

  ///Global Utility methods. Can always be used.

  int wrap(int index) const; //wrap: convert any index into a valid player index. For example if you do "yourIndex - 1", this gets converted to the index of the player left of you, even if yourIndex was 0

  int getPot() const; //sum of all players bets
  int getHighestWager() const; //highest amount of money put on the table by a player (including yourself). The call-amount can be calculated from this.

  int getNumPlayers() const; //amount of players at the table

  int getNumActivePlayers() const; //players that are not folded or out
  int getNumDecidingPlayers() const; //get amount of players that can still make decisions. All-in, folded or out players cannot.

  int getSmallBlind() const;
  int getBigBlind() const;
};


//TODO: generate info from events
//class InfoKeeper
//{
  //private:
    //Info info;

  //public:

    //const Info& getInfo() const;

    //InfoKeeper(int yourIndex); //set to -1 for global-only info
    //~InfoKeeper();

    //void onEvent(const Event& event);
//};
/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>

#include "action.h"
#include "event.h"

class AI;
struct Info;

/*
A Player is what joins the table and plays the game. Each player must have an AI,
which can be either a human or a bot.

For the rest, he has chips and a certain game status.

This class is used for running the Game, the AI's shouldn't use this class, they
get this information in the form of an Info struct instead.
*/
struct Player
{
  AI* ai; //the AI for the player

  int stack; //chips in his stack
  int wager; //how much chips this person currently has in the pot on the table (note: the "int stack" variable does NOT include these chips anymore, they're moved from stack to pot)

  int buyInTotal; //for how much money did this player buy in (used if rebuys are allowed to calculate score at end)

  Card holeCard1;
  Card holeCard2;

  bool folded;
  bool showdown; //this player (has to or wants to) show their cards

  std::string name;

  Action lastAction; //used for filling it in the Info

  Player(AI* ai, const std::string& name);

  void setCards(Card card1, Card card2);

  /*
  Rules about this name:
  -must have at least one character
  -max 12 characters, otherwise the ascii art is screwed up
  -spaces and dots are allowed
  -semicolons and commas are not allowed. This because semicolons are often used in logs and such, allowing parsers to know they're not part of a name.
  */
  std::string getName() const; //min 1 letter,
  std::string getAIName() const;

  Action doTurn(const Info& info);
  void onEvent(const Event& event);

  bool isAllIn() const;
  bool isOut() const; //can't play anymore, has no more money
  bool isFolded() const;

  bool isHuman() const;

  bool canDecide() const; //returns true if stack > 0 and not folded
};


std::string getRandomName();

struct PokerNetImpl : torch::nn::Module {
    // 1. Feature extraction layers
    torch::nn::Linear card_embedding{nullptr};
    torch::nn::LSTM rnn{nullptr};          // Processes the sequence of actions (history)
    torch::nn::Linear opponent_context{nullptr}; // Processes long-term opponent stats

    // 2. The "Vector" Head (Your custom geometry)
    torch::nn::Linear action_head{nullptr}; 

    PokerNetImpl(int input_size, int hidden_size) {
        // Embed cards/game state into a vector
        card_embedding = register_module("card_embed", torch::nn::Linear(input_size, 64));
        
        // LSTM takes (seq_len, batch, input_size)
        rnn = register_module("rnn", torch::nn::LSTM(torch::nn::LSTMOptions(64, hidden_size).num_layers(1)));
        
        // Compress opponent stats
        opponent_context = register_module("opp_ctx", torch::nn::Linear(10, 32)); // Assuming 10 stats

        // Output: 2 values (X, Y) for your vector regression
        action_head = register_module("action_head", torch::nn::Linear(hidden_size + 32, 2));
    }

    torch::Tensor forward(torch::Tensor game_state, torch::Tensor hidden_state, torch::Tensor opp_stats) {
        // 1. Process current game state
        auto x = torch::relu(card_embedding(game_state));
        
        // 2. Process history via LSTM
        // Note: In real impl, you manage the LSTM hidden tuple (h_n, c_n)
        auto rnn_out = rnn(x.unsqueeze(0)); 
        auto rnn_last_step = std::get<0>(rnn_out).squeeze(0);

        // 3. Process opponent context (Exploitative part)
        auto ctx = torch::relu(opponent_context(opp_stats));

        // 4. Combine (Concatenate)
        auto combined = torch::cat({rnn_last_step, ctx}, 1);

        // 5. Output Vector (x, y)
        return action_head(combined); 
    }
};

TORCH_MODULE(PokerNet);
/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>

#include "card.h"
#include "game.h"

/*
The table is where the games happen. This holds information used by the Game.
AI's can't use this information, they get an Info struct instead.
*/
struct Table
{
  std::vector<Player> players;
  std::vector<Observer*> observers;

  int dealer; //index of the dealer in the players vector
  int current; //index of the current player making a decision

  /*
  This is roughly the last person who raised. This is used to know when a betting round stops.
  This is made so that if the current player is the lastRaiser, the round ends.
  This takes the fact that the big blind can make a decision into account.
  */
  int lastRaiser;

  Round round;
  int turn; //how many decision making turns this round has had so far (if people keep raising all the time this could take forever!)

  int lastRaiseAmount; //last raise amount during this deal. This is used to disallow smaller raises. Initially this is set to the big blind. All-ins don't count towards this amount, so that it's possible to form a side-pot with smaller bets.

  //NOTE: the values of these cards are only valid if the Round is correct.
  //flop cards
  Card boardCard1;
  Card boardCard2;
  Card boardCard3;
  //turn card
  Card boardCard4;
  //river card
  Card boardCard5;

  Table();

  int getPot() const;
  int getHighestWager() const;
  int getCallAmount() const; //get amount of money required for you to call
  
  int getNumActivePlayers() const; //players that are not folded or out
  int getNumDecidingPlayers() const; //get amount of players that still make decision: players that aren't folded and aren't all-in

  int wrap(int index) const; //wrap: convert any index into a valid player index. For example if you do "yourIndex - 1", this gets converted to the index of the player left of you, even if yourIndex was 0
  int getSmallBlindIndex() const;
  int getBigBlindIndex() const;

  bool hasHumanPlayer() const;
  bool hadHumanPlayer() const; //the table has a human player now, or had one once but he's out
};
