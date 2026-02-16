#include "host_silent.h"
#include "io_terminal.h"

HostSilent::HostSilent()
: quit(false)
{
}

void HostSilent::onFrame()
{
  if(getCharNonBlocking() == 'q') quit = true;
}

void HostSilent::onGameBegin(const Info& info)
{
  (void)info;
}

void HostSilent::onDealDone(const Info& info)
{
  (void)info;
}

void HostSilent::onGameDone(const Info& info)
{
  (void)info;
}

bool HostSilent::wantToQuit() const
{
  return quit;
}

void HostSilent::resetWantToQuit()
{
  quit = false;
}
